import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from models.networks import ActorCriticNetwork

class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, actor_hidden_size=[128], critic_hidden_size=[128], gamma=0.99, lr=1e-3, update_every=10, use_gae=True, gae_lambda=0.95, episodes=1000, critic_coef=0.5, entropy_coef=0.01, ppo_epochs=10, clip_coef=0.2, batch_size=64):
        super().__init__(state_dim, action_dim, gamma, lr, update_every)
        
        self.model = ActorCriticNetwork(state_dim, action_dim, actor_hidden_size, critic_hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.05, total_iters=episodes)
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.clip_coef = clip_coef
        self.batch_size = batch_size

        # Buffers for the current rollout (now including dones to track episode boundaries)
        self.states, self.actions, self.rewards, self.log_probs, self.dones = [], [], [], [], []

    def select_action(self, state):
        """Select an action based on the current policy."""
        state = self.preprocess(state)
        logits = self.model.actor(state)
        # Create a categorical distribution from logits for numerical stability
        m = torch.distributions.Categorical(logits=logits)
        action = m.sample()
        
        # Only store experiences if the model is in training mode (prevents eval memory leak)
        if self.model.training:
            self.log_probs.append(m.log_prob(action))
            self.states.append(state) # Store state for critic update
            self.actions.append(action) # Store action for critic update
        return action.item()

    def collect_experience(self, state, action, reward, next_state, done):
        # PPO needs states, actions, rewards, and log_probs for its update
        # state, action, log_prob are already appended in select_action
        self.rewards.append(reward)
        self.dones.append(done)

    def try_update(self, next_state, done):
        # PPO only updates when the buffer has reached the required number of steps
        if len(self.rewards) < self.update_every:
            return
        
        # Perform the actual policy update
        self._perform_update_logic(next_state, done)

    def _perform_update_logic(self, next_state, done):
        # 1. Prepare Tensors from collected experience
        next_state = self.preprocess(next_state)
        with torch.no_grad():
            next_value = self.model.critic(next_state) if not done else torch.tensor([[0.0]]).to(self.device)
            
            s_t = torch.stack(self.states)
            v_t = self.model.critic(s_t).squeeze(-1)

            # 2. Compute Returns and Advantages using GAE
            if self.use_gae:
                advantages = self.compute_gae(self.rewards, v_t, next_value, self.dones)
                returns = advantages + v_t
            else: # Fallback to simple A2C-style returns/advantages
                returns_list = []
                R = next_value.item()
                for r, d in zip(reversed(self.rewards), reversed(self.dones)):
                    if d:
                        R = 0.0
                    R = r + self.gamma * R
                    returns_list.insert(0, R)
                returns = torch.tensor(returns_list, dtype=torch.float32).to(self.device)
                advantages = returns - v_t
        
        # Normalize advantages to stabilize training
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Detach tensors that are used as fixed targets in the PPO update loop
        b_states = s_t.detach()
        b_actions = torch.stack(self.actions).detach()
        b_log_probs_old = torch.stack(self.log_probs).detach()
        b_advantages = advantages.detach()
        b_returns = returns.detach()

        # 3. PPO Update Loop
        for _ in range(self.ppo_epochs):
            # Shuffle the batch indices to break correlation
            b_inds = torch.randperm(len(b_states), device=self.device)
            
            # Iterate over mini-batches
            for start in range(0, len(b_states), self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]
                
                # Re-compute policy distribution for the mini-batch states
                logits = self.model.actor(b_states[mb_inds])
                dist = Categorical(logits=logits)
                
                # Get new log_probs, entropy, and values for the mini-batch
                b_log_probs_new = dist.log_prob(b_actions[mb_inds])
                entropy_loss = dist.entropy().mean()
                b_values_new = self.model.critic(b_states[mb_inds]).squeeze(-1)

                # Calculate the ratio (pi_new / pi_old)
                log_ratio = b_log_probs_new - b_log_probs_old[mb_inds]
                ratio = torch.exp(log_ratio)

                # Calculate Actor Loss (Clipped Surrogate Objective)
                surr1 = ratio * b_advantages[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * b_advantages[mb_inds]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate Critic Loss
                critic_loss = nn.MSELoss()(b_values_new, b_returns[mb_inds])

                # Total Loss
                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_loss

                # 4. Step Optimizer
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        # 5. Clear buffers for the next rollout
        self.states, self.actions, self.rewards, self.log_probs, self.dones = [], [], [], [], []


    def compute_gae(self, rewards, values, next_value, dones):
        """Computes Generalized Advantage Estimation (GAE) for a trajectory"""
        advantages = torch.zeros(len(rewards), dtype=torch.float32).to(self.device)
        last_gae_lam = 0
        
        # Iterate backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = next_value.item()
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = values[t + 1].item()
                
            # TD-error for this step
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t].item()
            
            # Recursive GAE formula
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        return advantages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from models.networks import ActorCriticNetwork

class A2CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, actor_hidden_size=[128], critic_hidden_size=[128], gamma=0.99, lr=1e-3, update_every=10, use_gae=True, gae_lambda=0.95, episodes=1000, critic_coef=0.5, entropy_coef=0.05):
        super().__init__(state_dim, action_dim, gamma, lr, update_every)
        
        self.model = ActorCriticNetwork(state_dim, action_dim, actor_hidden_size, critic_hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.05, total_iters=episodes)
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        # Buffers for the current rollout
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
        # A2C needs states, actions, rewards, and log_probs for its update
        # state, action, log_prob are already appended in select_action
        self.rewards.append(reward)
        self.dones.append(done)

    def try_update(self, next_state, done):
        # A2C updates only when the buffer has reached the required number of steps
        if len(self.rewards) < self.update_every:
            return
        
        # Perform the actual policy update
        self._perform_update_logic(next_state, done)

    def _perform_update_logic(self, next_state, done): # Renamed to internal method
        # 1. Prepare Tensors
        # We need the value of the next_state to "bootstrap" the return
        next_state = self.preprocess(next_state)
        next_value = self.model.critic(next_state).detach() if not done else torch.tensor([[0.0]]).to(self.device)
        
        # Stack buffers into tensors
        s_t = torch.stack(self.states)
        a_t = torch.stack(self.actions)
        lp_t = torch.stack(self.log_probs)
        v_t = self.model.critic(s_t).squeeze(-1) # Predicted values for these states

        # 2. Compute Returns and Advantages
        # Standard A2C calculates returns backwards step-by-step
        returns = []
        R = next_value.item()
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device) # Ensure float32
        
        # Advantage = Actual Return - Predicted Value
        if self.use_gae:
            # If using GAE, we compute advantages differently
            advantages = self.compute_gae(self.rewards, v_t, next_value, self.dones)
            returns = advantages + v_t.detach() # Update returns to TD(lambda) targets
        else:
            advantages = (returns - v_t).detach()

        # Normalize advantages to stabilize training
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Calculate Losses
        # Actor Loss: -log_prob * Advantage
        actor_loss = -(lp_t * advantages).mean()
        
        # Critic Loss: MSE between predicted value and actual return
        critic_loss = nn.MSELoss()(v_t, returns)
        
        # Entropy Loss: Use a distribution to get entropy from the current policy head
        logits = self.model.actor(s_t)
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()

        # Total Loss: Weighted sum
        # Typical weights: Critic=0.5, Entropy=0.01
        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

        # 4. Step Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        # turns up that gradient clipping is vital for A2C stability
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # 5. Clear buffers
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
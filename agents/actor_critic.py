import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from models.networks import ActorCriticNetwork

class A2CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, actor_hidden_size = [128], critic_hidden_size=[128], gamma=0.99, lr=1e-3, update_every=10, use_gae=True, gae_lambda=0.95):
        super().__init__(state_dim, action_dim, gamma, lr, update_every)
        
        self.model = ActorCriticNetwork(state_dim, action_dim, actor_hidden_size, critic_hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        # Buffers for the current rollout
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []

    def select_action(self, state):
        """Select an action based on the current policy."""
        state = self.preprocess(state)
        probs = self.model.actor(state)
        # Create a categorical distribution to sample steering/speed actions
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.states.append(state) # Store state for critic update
        self.actions.append(action) # Store action for critic update
        return action.item()

    def collect_experience(self, state, action, reward, next_state, done):
        # A2C needs states, actions, rewards, and log_probs for its update
        # state, action, log_prob are already appended in select_action
        self.rewards.append(reward)

    def try_update(self, next_state, done):
        # A2C updates if enough steps have been collected or if the episode ends
        if len(self.rewards) < self.update_every and not done:
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
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device) # Ensure float32
        
        # Advantage = Actual Return - Predicted Value
        if self.use_gae:
            # If using GAE, we compute advantages differently
            advantages = self.compute_gae(self.rewards, v_t, next_value, done)
            returns = advantages + v_t.detach() # Update returns to TD(lambda) targets
        else:
            advantages = (returns - v_t).detach()


        # 3. Calculate Losses
        # Actor Loss: -log_prob * Advantage
        actor_loss = -(lp_t * advantages).mean()
        
        # Critic Loss: MSE between predicted value and actual return
        critic_loss = nn.MSELoss()(v_t, returns)
        
        # Entropy Loss: Use a distribution to get entropy from the current policy head
        probs = self.model.actor(s_t)
        dist = Categorical(probs)
        entropy = dist.entropy().mean()

        # Total Loss: Weighted sum
        # Typical weights: Critic=0.5, Entropy=0.01
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # 4. Step Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        # turns up that gradient clipping is vital for A2C stability
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # 5. Clear buffers
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []


    def compute_gae(self, rewards, values, next_value, done):
        """Computes Generalized Advantage Estimation (GAE) for a trajectory"""
        advantages = torch.zeros(len(rewards), dtype=torch.float32).to(self.device)
        last_gae_lam = 0
        
        # Iterate backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(done)
                next_val = next_value.item()
            else:
                next_non_terminal = 1.0
                next_val = values[t + 1].item()
                
            # TD-error for this step
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t].item()
            
            # Recursive GAE formula
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        return advantages
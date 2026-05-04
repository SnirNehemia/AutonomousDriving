import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from models.networks import MLPNetwork

class REINFORCEAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, hidden_size=[128], gamma=0.99, lr=1e-3, update_every=1, episodes=1000):
        super().__init__(state_dim, action_dim, gamma, lr, update_every)
        
        self.model = MLPNetwork(state_dim, action_dim, hidden_size, use_softmax=False, is_actor=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.05, total_iters=episodes)
        
        # Buffers for the current rollout (specific to REINFORCE)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """Select an action based on the current policy."""
        state = self.preprocess(state)
        logits = self.model(state)
        # Create a categorical distribution from logits for numerical stability
        m = torch.distributions.Categorical(logits=logits)
        action = m.sample()
        
        # Only store experiences if the model is in training mode
        if self.model.training:
            self.log_probs.append(m.log_prob(action))
        return action.item()

    def collect_experience(self, state, action, reward, next_state, done):
        # REINFORCE only needs rewards and log_probs for its update
        self.rewards.append(reward)
        # log_prob is already appended in select_action

    def try_update(self, next_state=None, done=True): # next_state is not used by REINFORCE but kept for signature
        # REINFORCE only updates at the end of an episode (Monte Carlo)
        if not done:
            return

        # Compute discounted returns G_t using montecarlo method
        returns = []
        G_t = 0
        for r in reversed(self.rewards):
            G_t = r + self.gamma * G_t
            returns.insert(0, G_t)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device) # Ensure float32
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Normalize returns
        # Calculate loss: mean(-log_prob * G_t) to avoid scaling by episode length
        log_probs_tensor = torch.stack(self.log_probs)
        loss = -torch.mean(log_probs_tensor * returns)

            
        # Perform backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Clear stored values
        self.log_probs = []
        self.rewards = []
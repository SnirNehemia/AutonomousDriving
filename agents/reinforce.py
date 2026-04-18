import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import OmegaConf

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, hidden_size = [128], gamma=0.99, lr=1e-3):
        layers = []
        input_dim = state_dim
        
        # Dynamically create hidden layers
        for h_dim in hidden_size:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim # Output of current layer becomes input for next
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.policy = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.gamma = gamma

    def select_action(self, state):
        """Select an action based on the current policy."""
        state = torch.from_numpy(state).float()
        probs = self.policy(state)
        # Create a categorical distribution to sample steering/speed actions
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        # Compute discounted returns G_t using montecarlo method
        returns = []
        G_t = 0
        for r in reversed(self.rewards):
            G_t = r + self.gamma * G_t
            returns.insert(0, G_t)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Normalize returns
        # Calculate loss: -log_prob * G_t
        log_probs_tensor = torch.stack(self.log_probs)
        loss = -torch.sum(log_probs_tensor * returns)

            
        # Perform backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Clear stored values
        self.log_probs = []
        self.rewards = []
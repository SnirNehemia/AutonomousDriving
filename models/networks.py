import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=[128], use_softmax=False):
        super().__init__()
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_size:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim 
            
        layers.append(nn.Linear(curr_dim, output_dim))
        if use_softmax:
            layers.append(nn.Softmax(dim=-1))
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_size=[128], critic_hidden_size=[128]):
        super().__init__()
        # The Actor outputs action probabilities, the Critic outputs a single state value
        self.actor = MLPNetwork(state_dim, action_dim, actor_hidden_size, use_softmax=True)
        self.critic = MLPNetwork(state_dim, 1, critic_hidden_size, use_softmax=False)
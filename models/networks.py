import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=[128], use_softmax=False, is_actor=False):
        super().__init__()
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_size:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim 
            
        last_layer = nn.Linear(curr_dim, output_dim)
        
        # If this is a policy head, scale down the initial weights so the agent 
        # starts with a near-uniform distribution, ensuring maximum early exploration.
        if is_actor:
            nn.init.orthogonal_(last_layer.weight, gain=0.01)
            nn.init.constant_(last_layer.bias, 0.0)
            
        layers.append(last_layer)
        if use_softmax:
            layers.append(nn.Softmax(dim=-1))
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_size=[128], critic_hidden_size=[128]):
        super().__init__()
        # The Actor outputs action probabilities, the Critic outputs a single state value
        self.actor = MLPNetwork(state_dim, action_dim, actor_hidden_size, use_softmax=False, is_actor=True)
        self.critic = MLPNetwork(state_dim, 1, critic_hidden_size, use_softmax=False)
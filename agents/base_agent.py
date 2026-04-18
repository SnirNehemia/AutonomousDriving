import torch
import os

class BaseAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # These will be initialized by the child classes
        self.model = None
        self.optimizer = None
        
        # Shared storage for training metrics
        self.stats = {"loss": [], "entropy": []}
        self.rewards = []
        self.log_probs = []

    def preprocess(self, state):
        """Standardizes input for the NN."""
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        return state.flatten()

    def save(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        # We save the model AND the optimizer state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle backwards compatibility with older state-dict-only saves
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
    def select_action(self, state):
        raise NotImplementedError("Agents must implement select_action method.")
        
    def update_policy(self):
        raise NotImplementedError("Agents must implement update_policy method.")
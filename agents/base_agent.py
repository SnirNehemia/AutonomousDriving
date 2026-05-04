import torch
import os

class BaseAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, update_every=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # These will be initialized by the child classes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Shared storage for training metrics
        self.stats = {"loss": [], "entropy": []}
        
    def preprocess(self, state):
        """Standardizes input for the NN."""
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        return state.flatten()

    def save(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        # We save the model, optimizer, and scheduler states
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle backwards compatibility with older state-dict-only saves
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
    def select_action(self, state):
        raise NotImplementedError("Agents must implement select_action method.")
        
    def collect_experience(self, state, action, reward, next_state, done):
        raise NotImplementedError("Agents must implement collect_experience method.")

    def try_update(self, next_state, done):
        raise NotImplementedError("Agents must implement update_policy method.")

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
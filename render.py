import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import highway_env
from omegaconf import OmegaConf
import glob
import numpy as np
from PIL import Image, ImageDraw

from agents.reinforce import REINFORCEAgent
from agents.actor_critic import A2CAgent

class CustomVisualizerWrapper(gym.Wrapper):
    def __init__(self, env, agent):
        super().__init__(env)
        self.agent = agent
        self.current_obs = None
        self.total_reward = 0.0
        self.step_counter = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_obs = obs
        # Discount the reward using gamma raised to the current step power
        self.total_reward += reward * (self.agent.gamma ** self.step_counter)
        self.step_counter += 1
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        self.total_reward = 0.0
        self.step_counter = 0
        return obs, info
        
    def _draw_network_vis(self, draw, activations, title, x_start, y_start, width):
        """Helper function to draw a single network visualization block."""
        draw.text((x_start, y_start), title, fill=(220, 220, 220))
        
        for i, act in enumerate(activations):
            y_offset = y_start + 25 + i * 65
            
            if i == 0: label = "Input"
            elif i == len(activations) - 1: label = "Output"
            else: label = f"Hidden {i}"
                
            draw.text((x_start, y_offset - 15), label, fill=(200, 200, 200))
            
            num_nodes = len(act)
            node_w = min(12.0, (width - 20) / max(1, num_nodes))
            spacing = (width - 20 - (num_nodes * node_w)) / max(1, num_nodes) if num_nodes > 1 else 0
            
            for j, val in enumerate(act):
                x0 = x_start + j * (node_w + spacing)
                x1 = x0 + node_w
                y0 = y_offset
                y1 = y0 + 20
                
                # Coloring rules
                if i == 0: v = max(0, min(1, (val + 1) / 2))
                elif i == len(activations) - 1: v = max(0, min(1.0, val))
                else: v = max(0, min(1.0, val / 3.0))
                    
                r = int(v * 255)
                b = int((1 - v) * 255)
                draw.rectangle([x0, y0, x1, y1], fill=(r, 50, b), outline=(20, 20, 20))

    def render(self):
        frame = self.env.render()
        if frame is None: return None
            
        img = Image.fromarray(frame)
        W, H = img.size
        top_pad = 40
        
        # 1. Get network activations based on agent type
        acts_reinforce, acts_actor, acts_critic = [], [], []
        num_layers = 0
        if self.current_obs is not None:
            flat_obs = self.current_obs.flatten()
            with torch.no_grad():
                x = self.agent.preprocess(flat_obs)
                if isinstance(self.agent, REINFORCEAgent):
                    acts_reinforce.append(x.cpu().numpy())
                    for layer in self.agent.model.net:
                        x = layer(x)
                        if isinstance(layer, (torch.nn.ReLU, torch.nn.Softmax)):
                            acts_reinforce.append(x.cpu().numpy())
                    num_layers = len(acts_reinforce)
                elif isinstance(self.agent, A2CAgent):
                    # Actor
                    x_actor = x.clone()
                    acts_actor.append(x_actor.cpu().numpy())
                    for layer in self.agent.model.actor.net:
                        x_actor = layer(x_actor)
                        if isinstance(layer, (torch.nn.ReLU, torch.nn.Softmax)):
                            acts_actor.append(x_actor.cpu().numpy())
                    # Critic
                    x_critic = x.clone()
                    acts_critic.append(x_critic.cpu().numpy())
                    for layer in self.agent.model.critic.net:
                        x_critic = layer(x_critic)
                        if isinstance(layer, torch.nn.ReLU): # Critic has no final softmax
                            acts_critic.append(x_critic.cpu().numpy())
                    acts_critic.append(x_critic.cpu().numpy()) # Add final output layer
                    num_layers = max(len(acts_actor), len(acts_critic))

        # 2. Create the new canvas
        bottom_pad = 40 + num_layers * 65 if num_layers > 0 else 0
        new_img = Image.new('RGB', (W, H + top_pad + bottom_pad), (40, 40, 40))
        new_img.paste(img, (0, top_pad))
        draw = ImageDraw.Draw(new_img)
        
        # 3. Draw header text
        try: speed, distance = self.env.unwrapped.vehicle.speed, self.env.unwrapped.vehicle.position[0]
        except: speed, distance = 0.0, 0.0
        text = f"Discounted Score: {self.total_reward:.2f} | Speed: {speed:.1f} m/s | Distance: {distance:.1f} m"
        draw.text((20, 10), text, fill=(255, 255, 255))
                        
        # 4. Draw network visualizations
        if acts_reinforce:
            self._draw_network_vis(draw, acts_reinforce, "REINFORCE Network", 20, H + top_pad + 20, W - 40)
        elif acts_actor:
            self._draw_network_vis(draw, acts_actor, "Actor Network", 20, H + top_pad + 20, W/2 - 30)
            self._draw_network_vis(draw, acts_critic, "Critic Network", W/2 + 10, H + top_pad + 20, W/2 - 30)
        
        return np.array(new_img)

def get_latest_run_dir(base_path="results"):
    """Finds the latest run directory based on modification time."""
    list_of_dirs = [d for d in glob.glob(f'{base_path}/*') if os.path.isdir(d)]
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

def render():
    # 1. Load render configuration
    render_config = OmegaConf.load("config_render.yaml")

    # 2. Determine the run directory
    if render_config.use_recent:
        run_path = get_latest_run_dir()
        if run_path is None:
            print("No recent run found in 'results/' directory.")
            return
    else:
        run_id = f"{render_config.project.version}_{render_config.project.run_name}"
        run_path = f"results/{run_id}"

    if not os.path.isdir(run_path):
        print(f"Run directory not found: {run_path}")
        return

    print(f"Rendering agents from: {run_path}")

    # 3. Load training configuration
    train_config_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(train_config_path):
        print(f"Training config 'config.yaml' not found in {run_path}")
        return
    train_config = OmegaConf.load(train_config_path)

    # 4. Get environment and agent dimensions
    try:
        # Use env_config from the run if it exists, otherwise use an empty dict
        env_config_dict = {}
        if 'env_config' in train_config:
            env_config_dict = OmegaConf.to_container(train_config.env_config, resolve=True)

        temp_env = gym.make(train_config.env.id, config=env_config_dict)
        state_dim = np.prod(temp_env.observation_space.shape)
        action_dim = temp_env.action_space.n
        temp_env.close()
    except Exception as e:
        print(f"Error creating temp env to get dimensions: {e}")
        return

    # 5. Find all model files
    model_paths = glob.glob(os.path.join(run_path, "*.pth"))
    if not model_paths:
        print(f"No model files (.pth) found in {run_path}")
        return
    
    print(f"Found models: {[os.path.basename(p) for p in model_paths]}")

    # 6. Iterate through models and seeds to render videos
    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\n--- Rendering for model: {model_name} ---")

        # Create agent and load weights
        agent_type = train_config.agent.type
        print(f"  Instantiating agent of type: {agent_type}")
        
        if agent_type == "REINFORCEAgent":
            agent_hidden_size = train_config.agent.get("hidden_size", [128])
            agent = REINFORCEAgent(state_dim, action_dim, hidden_size=agent_hidden_size, lr=train_config.agent.lr, gamma=train_config.agent.gamma)
        elif agent_type == "A2CAgent":
            actor_hidden_size = train_config.agent.get("actor_hidden_size", [128])
            critic_hidden_size = train_config.agent.get("critic_hidden_size", [128])
            agent = A2CAgent(state_dim, action_dim, actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size, lr=train_config.agent.lr, gamma=train_config.agent.gamma)
        else:
            print(f"  Unknown agent type '{agent_type}' in config. Skipping model.")
            continue

        agent.load(model_path)
        agent.model.eval() # Set policy to evaluation mode

        for seed in render_config.seeds_to_run:
            print(f"  Running with seed: {seed}")
            
            # Fresh config for each seed to prevent policy_frequency override leak
            run_env_config = dict(env_config_dict)
            sim_freq = run_env_config.get('simulation_frequency', 15)
            orig_policy_freq = run_env_config.get('policy_frequency', 1)
            
            # Force policy frequency to match simulation frequency to get intermediate frames
            run_env_config['policy_frequency'] = sim_freq
            steps_per_action = max(1, sim_freq // orig_policy_freq)

            # Create environment
            env = gym.make(train_config.env.id, config=run_env_config, render_mode="rgb_array")
            
            # Wrap with our custom visualizer
            env = CustomVisualizerWrapper(env, agent)
            
            # Setup video recording
            video_folder = os.path.join(run_path, "rendered_videos", model_name)
            os.makedirs(video_folder, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix=f"seed_{seed}",
                episode_trigger=lambda e: True # Record all episodes run
            )

            obs, info = env.reset(seed=seed)
            obs = obs.flatten()
            done = truncated = False
            total_reward = 0
            step_count = 0
            current_action = 0

            while not (done or truncated):
                # Sample action at the original policy frequency
                if step_count % steps_per_action == 0:
                    with torch.no_grad():
                        current_action = agent.select_action(obs)
                        
                obs, reward, terminated, truncated, info = env.step(current_action)
                obs = obs.flatten()
                total_reward += reward
                done = terminated
                step_count += 1

            print(f"\tSeed {seed} finished. Total reward: {total_reward:.2f}.")
            print(f"\tVideo saved to {video_folder}")
            env.close()

if __name__ == "__main__":
    render()
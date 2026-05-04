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
from agents.ppo import PPOAgent

class CustomVisualizerWrapper(gym.Wrapper):
    def __init__(self, env, agent):
        super().__init__(env)
        self.agent = agent
        self.current_obs = None
        self.total_reward = 0.0
        self.step_counter = 0
        self.last_action = None
        
    def step(self, action):
        self.last_action = action
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
        self.last_action = None
        return obs, info
        
    def _draw_network_vis(self, draw, activations, title, x_start, y_start, width, is_actor=False, is_critic=False):
        """Helper function to draw a single network visualization block."""
        draw.text((x_start, y_start), title, fill=(220, 220, 220))
        
        for i, act in enumerate(activations):
            y_offset = y_start + 25 + i * 65
            
            is_output = (i == len(activations) - 1)
            if i == 0: label = "Input"
            elif is_output: label = "Output"
            else: label = f"Hidden {i}"
                
            draw.text((x_start, y_offset - 15), label, fill=(200, 200, 200))
            
            num_nodes = len(act)
            node_w = min(12.0, (width - 20) / max(1, num_nodes))
            spacing = (width - 20 - (num_nodes * node_w)) / max(1, num_nodes) if num_nodes > 1 else 0
            
            # Dynamically compute max value for hidden layers to prevent color saturation
            layer_max = max(1.0, float(np.max(np.abs(act)))) if not is_output else 1.0
            
            for j, val in enumerate(act):
                x0 = x_start + j * (node_w + spacing)
                x1 = x0 + node_w
                y0 = y_offset
                y1 = y0 + 20
                
                # Coloring rules
                if i == 0: 
                    v = max(0, min(1, (val * 3.0 + 1) / 2))
                elif is_output:
                    if is_critic:
                        v = max(0, min(1.0, val / 50.0)) # Scale expected value bounds for Critic
                    else:
                        v = max(0, min(1.0, val)) # Softmax probabilities for Actor
                else: 
                    v = max(0, min(1.0, val / layer_max)) # Normalize hidden layers dynamically
                    
                r = int(v * 255)
                b = int((1 - v) * 255)
                
                # Highlight the action the agent actually chose with a bright green outline
                is_chosen_action = (is_output and is_actor and self.last_action is not None and j == self.last_action)
                outline_color = (0, 255, 0) if is_chosen_action else (20, 20, 20)
                
                draw.rectangle([x0, y0, x1, y1], fill=(r, 50, b), outline=outline_color)
                if is_chosen_action:
                    draw.rectangle([x0+1, y0+1, x1-1, y1-1], outline=outline_color) # Inner highlight

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
                        if isinstance(layer, torch.nn.ReLU):
                            acts_reinforce.append(x.cpu().numpy())
                    acts_reinforce.append(torch.softmax(x, dim=-1).cpu().numpy())
                    num_layers = len(acts_reinforce)
                elif isinstance(self.agent, (A2CAgent, PPOAgent)):
                    # Actor
                    x_actor = x.clone()
                    acts_actor.append(x_actor.cpu().numpy())
                    for layer in self.agent.model.actor.net:
                        x_actor = layer(x_actor)
                        if isinstance(layer, torch.nn.ReLU):
                            acts_actor.append(x_actor.cpu().numpy())
                    acts_actor.append(torch.softmax(x_actor, dim=-1).cpu().numpy())
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
            self._draw_network_vis(draw, acts_reinforce, "REINFORCE Network", 20, H + top_pad + 20, W - 40, is_actor=True)
        elif acts_actor:
            self._draw_network_vis(draw, acts_actor, "Actor Network", 20, H + top_pad + 20, W/2 - 30, is_actor=True)
            self._draw_network_vis(draw, acts_critic, "Critic Network", W/2 + 10, H + top_pad + 20, W/2 - 30, is_critic=True)
        
        return np.array(new_img)

def get_latest_run_dir(base_path="results"):
    """Finds the latest run directory based on modification time."""
    list_of_dirs = [d for d in glob.glob(f'{base_path}/*') if os.path.isdir(d)]
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

def render(base_path=None, seeds_to_run=None):
    if base_path is None or seeds_to_run is None:
        # 1. Load render configuration
        try:
            render_config = OmegaConf.load("config_render.yaml")
        except Exception as e:
            print(f"Could not load config_render.yaml: {e}")
            return

        # 2. Determine the run directory
        if render_config.use_recent:
            base_path = get_latest_run_dir()
            if base_path is None:
                print("No recent run found in 'results/' directory.")
                return
        else:
            run_id = f"{render_config.project.version}_{render_config.project.run_name}"
            possible_dirs = glob.glob(f"results/{run_id}_*")
            if possible_dirs:
                base_path = possible_dirs[0]
            else:
                base_path = f"results/{run_id}_single"
            
        seeds_to_run = render_config.seeds_to_run

    if not os.path.isdir(base_path):
        print(f"Run directory not found: {base_path}")
        return

    print(f"Rendering agents from: {base_path}")

    # 3. Load training configuration
    train_config_path = os.path.join(base_path, "config.yaml")
    if not os.path.exists(train_config_path):
        print(f"Training config 'config.yaml' not found in {base_path}")
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
    model_paths = glob.glob(os.path.join(base_path, "**", "models", "*.pth"), recursive=True)
    if not model_paths:
        print(f"No model files (.pth) found in {base_path}")
        return
    
    print(f"Found models: {[os.path.basename(p) for p in model_paths]}")

    # 6. Iterate through models and seeds to render videos
    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\n--- Rendering for model: {model_name} ---")

        # Determine agent configuration based on mode
        agent_config = None
        if "agent" in train_config:
            agent_config = train_config.agent
        elif train_config.get("mode") == "single":
            agent_config = train_config.single_run.agent
        elif train_config.get("mode") == "ablation":
            # Extract experiment name from the directory structure
            # e.g. path is .../<exp_name>_seed_<seed>/models/model.pth
            seed_dir_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            exp_name = seed_dir_name.split("_seed_")[0]
            for exp in train_config.ablation.experiments:
                if exp.name == exp_name:
                    # Reconstruct the full agent config by merging single_run.agent and experiment overrides
                    agent_config = OmegaConf.create()
                    if "single_run" in train_config and "agent" in train_config.single_run:
                        agent_config = OmegaConf.merge(agent_config, train_config.single_run.agent)
                    if "agent" in exp:
                        agent_config = OmegaConf.merge(agent_config, exp.agent)
                    break
                    
        if agent_config is None:
            print(f"  Could not determine agent configuration for {model_name}. Skipping.")
            continue

        agent_type = agent_config.type
        print(f"  Instantiating agent of type: {agent_type}")
        
        if agent_type == "REINFORCEAgent":
            agent_hidden_size = agent_config.get("hidden_size", [128])
            agent = REINFORCEAgent(state_dim, action_dim, hidden_size=agent_hidden_size, lr=agent_config.lr, gamma=agent_config.gamma)
        elif agent_type == "A2CAgent":
            actor_hidden_size = agent_config.get("actor_hidden_size", [128])
            critic_hidden_size = agent_config.get("critic_hidden_size", [128])
            agent = A2CAgent(state_dim, action_dim, actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                             use_gae=agent_config.use_gae, gae_lambda=agent_config.gae_lambda,
                               lr=agent_config.lr, gamma=agent_config.gamma,
                               critic_coef=agent_config.get("critic_coef", 0.5), entropy_coef=agent_config.get("entropy_coef", 0.01))
        elif agent_type == "PPOAgent":
            actor_hidden_size = agent_config.get("actor_hidden_size", [128])
            critic_hidden_size = agent_config.get("critic_hidden_size", [128])
            agent = PPOAgent(state_dim, action_dim, actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                             use_gae=agent_config.use_gae, gae_lambda=agent_config.gae_lambda,
                               lr=agent_config.lr, gamma=agent_config.gamma,
                               critic_coef=agent_config.get("critic_coef", 0.5), entropy_coef=agent_config.get("entropy_coef", 0.01),
                               ppo_epochs=agent_config.get("ppo_epochs", 10), clip_coef=agent_config.get("clip_coef", 0.2),
                               batch_size=agent_config.get("batch_size", 64))
        else:
            print(f"  Unknown agent type '{agent_type}' in config. Skipping model.")
            continue

        agent.load(model_path)
        agent.model.eval() # Set policy to evaluation mode

        for seed in seeds_to_run:
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
            models_dir = os.path.dirname(model_path)
            seed_dir = os.path.dirname(models_dir)
            seed_dir_name = os.path.basename(seed_dir)
            video_folder = os.path.join(base_path, "videos", seed_dir_name)
            os.makedirs(video_folder, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix=f"{model_name}_testseed_{seed}",
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
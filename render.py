import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import highway_env
from omegaconf import OmegaConf
import glob
import numpy as np

from agents.reinforce import REINFORCEAgent

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
        agent_hidden_size = train_config.agent.get("hidden_size", [128]) # Get hidden_size from train_config, default to [128] for backward compatibility
        agent = REINFORCEAgent(state_dim, action_dim, hidden_size=agent_hidden_size, lr=train_config.agent.lr, gamma=train_config.agent.gamma)
        agent.policy.load_state_dict(torch.load(model_path))
        agent.policy.eval() # Set policy to evaluation mode

        for seed in render_config.seeds_to_run:
            print(f"  Running with seed: {seed}")
            
            # Create environment
            env = gym.make(train_config.env.id, config=env_config_dict, render_mode="rgb_array")
            
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

            while not (done or truncated):
                with torch.no_grad():
                    action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = obs.flatten()
                total_reward += reward
                done = terminated

            print(f"\tSeed {seed} finished. Total reward: {total_reward:.2f}.")
            print(f"\tVideo saved to {video_folder}")
            env.close()

if __name__ == "__main__":
    render()
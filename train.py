import os
import gymnasium as gym
import torch
import highway_env # Import to register highway environments with Gymnasium
import matplotlib.pyplot as plt
import numpy as np
import agents.ppo
import agents.reinforce
import agents.actor_critic
from omegaconf import OmegaConf
import time

def smooth_curve(scores, window=10):
    """Computes a rolling average to smooth the curve."""
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(scores[start:i+1]))
    return np.array(smoothed)

def train_single_run(config, seed, models_path):
    """Trains a single agent configuration for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Create Environment
    env_config_dict = OmegaConf.to_container(config.env_config, resolve=True)
    env = gym.make(config.env.id, config=env_config_dict, render_mode=config.env.render_mode)

    # 3. Create Agent
    # Get state and action dimensions from the environment
    obs_space_shape = env.observation_space.shape
    state_dim = np.prod(obs_space_shape)
    action_dim = env.action_space.n
    
    if config.agent.type == "REINFORCEAgent":
        agent = agents.reinforce.REINFORCEAgent(state_dim, action_dim, hidden_size=config.agent.hidden_size,
                                lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every,
                                episodes=config.agent.episodes)
    elif config.agent.type == "A2CAgent":
        agent = agents.actor_critic.A2CAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size,
                                             use_gae=config.agent.use_gae, gae_lambda=config.agent.gae_lambda,
                                  lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every,
                                  episodes=config.agent.episodes,
                                  critic_coef=config.agent.get("critic_coef", 0.5),
                                  entropy_coef=config.agent.get("entropy_coef", 0.01))
    elif config.agent.type == "PPOAgent":
        agent = agents.ppo.PPOAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size,
                                  use_gae=config.agent.use_gae, gae_lambda=config.agent.gae_lambda,
                                  lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every,
                                  episodes=config.agent.episodes,
                                  critic_coef=config.agent.get("critic_coef", 0.5), entropy_coef=config.agent.get("entropy_coef", 0.01),
                                  ppo_epochs=config.agent.get("ppo_epochs", 10), clip_coef=config.agent.get("clip_coef", 0.2),
                                  batch_size=config.agent.get("batch_size", 64))
    else:
        raise ValueError(f"Unknown agent type: {config.agent.type}")

    scores = []
    t_start = time.time()

    # 4. Training Loop
    for ep in range(config.agent.episodes):
        # Vary the seed per episode to prevent overfitting to a single scenario
        obs, info = env.reset(seed=seed * 10000 + ep)
        obs = obs.flatten()
        done = truncated = False
        ep_reward = 0

        while not (done or truncated):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.collect_experience(obs, action, reward, next_obs.flatten(), terminated or truncated)
            obs = next_obs.flatten()
            ep_reward += reward
            done = terminated or truncated
            agent.try_update(obs, done)
            
        agent.step_scheduler()
        scores.append(ep_reward)
        if (ep + 1) % (10 if config.mode == "single" else 50) == 0:
            print(f"  Episode {ep+1}/{config.agent.episodes} - Score: {ep_reward:.2f} - time: {(time.time() - t_start) / 60:.1f} min")
        if ep > 0 and ep % config.agent.save_agent_every == 0:
            agent.save(models_path, f"model_ep{ep}.pth")

    # 4. Save Final Model and Cleanup
    final_model_path = os.path.join(models_path, "model_final.pth")
    agent.save(models_path, "model_final.pth")
    env.close()
    return scores, final_model_path


def test_single_model(model_path, config, test_seeds):
    """Evaluates a trained model on a set of test seeds."""
    print(f"--- Testing model: {model_path} ---")
    test_scores = []

    # Create a temporary env to get dimensions
    env_config_dict = OmegaConf.to_container(config.env_config, resolve=True)
    temp_env = gym.make(config.env.id, config=env_config_dict)
    state_dim = np.prod(temp_env.observation_space.shape)
    action_dim = temp_env.action_space.n
    temp_env.close()

    # Instantiate the correct agent
    if config.agent.type == "REINFORCEAgent":
        agent = agents.reinforce.REINFORCEAgent(state_dim, action_dim, hidden_size=config.agent.hidden_size)
    elif config.agent.type == "A2CAgent":
        agent = agents.actor_critic.A2CAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size,
                                             use_gae=config.agent.use_gae, gae_lambda=config.agent.gae_lambda,
                                             critic_coef=config.agent.get("critic_coef", 0.5),
                                             entropy_coef=config.agent.get("entropy_coef", 0.05))
    elif config.agent.type == "PPOAgent":
        agent = agents.ppo.PPOAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size,
                                  use_gae=config.agent.use_gae, gae_lambda=config.agent.gae_lambda,
                                  critic_coef=config.agent.get("critic_coef", 0.5), entropy_coef=config.agent.get("entropy_coef", 0.01),
                                  ppo_epochs=config.agent.get("ppo_epochs", 10), clip_coef=config.agent.get("clip_coef", 0.2),
                                  batch_size=config.agent.get("batch_size", 64))
    else:
        raise ValueError(f"Unknown agent type: {config.agent.type}")

    agent.load(model_path)
    agent.model.eval()

    env = gym.make(config.env.id, config=env_config_dict, render_mode="rgb_array")
    for seed in test_seeds:
        obs, info = env.reset(seed=seed)
        obs = obs.flatten()
        done = truncated = False
        ep_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs.flatten()
            ep_reward += reward
            done = terminated or truncated
        test_scores.append(ep_reward)
        print(f"  Test Seed {seed}: Score = {ep_reward:.2f}")
    env.close()
    return test_scores


def plot_ablation_results(all_results, ablation_path):
    """Plots the mean and std dev of training scores for all experiments."""
    plt.figure(figsize=(12, 8)) 
    for name, results in all_results.items():
        scores_np = np.array(results['train_scores'])
        mean_scores = np.mean(scores_np, axis=0)
        std_scores = np.std(scores_np, axis=0)
        smoothed_mean = smooth_curve(mean_scores, window=10)
        
        plt.plot(smoothed_mean, label=name)
        plt.fill_between(range(len(smoothed_mean)), smoothed_mean - std_scores, smoothed_mean + std_scores, alpha=0.2)

    plt.title("Ablation Study: Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ablation_path, "ablation_training_plot.png"))
    plt.close()
    print(f"\nAblation plot saved to {os.path.join(ablation_path, 'ablation_training_plot.png')}")

def run_ablation_study(config):
    """Runs the ablation study based on the unified config."""
    t_start = time.time()
    
    base_dir_name = f"{config.project.version}_{config.project.run_name}_{config.mode}"
    base_path = os.path.join("results", base_dir_name)
    os.makedirs(base_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(base_path, "config.yaml"))
    
    all_results = {}
    
    print(f"=== Starting Ablation Study: {config.project.run_name} ===")
    
    for exp_config in config.ablation.experiments:
        exp_name = exp_config.name
        print(f"\n--- Running Experiment: {exp_name} ---")
        
        # Create a deep copy of the base config
        full_config = OmegaConf.merge(config)
        
        # Inject base agent from single_run, so experiments can just override specific parameters
        if "single_run" in full_config and "agent" in full_config.single_run:
            full_config.agent = full_config.single_run.agent
            
        # Merge experiment config (deep merges the 'agent' block, overriding base parameters)
        full_config = OmegaConf.merge(full_config, exp_config)
        
        exp_train_scores = []
        trained_model_paths = []
        
        for seed in config.seeds.train:
            print(f"  Training with seed: {seed}")
            seed_dir = os.path.join(base_path, f"{exp_name}_seed_{seed}")
            models_dir = os.path.join(seed_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            scores, final_model_path = train_single_run(full_config, seed, models_dir)
            exp_train_scores.append(scores)
            trained_model_paths.append(final_model_path)
        
        # For simplicity, we test the model from the first training seed
        # A more robust approach could be to test all models and average the results
        model_to_test = trained_model_paths[0]
        exp_test_scores = test_single_model(model_to_test, full_config, config.seeds.test)
        
        all_results[exp_name] = {
            'train_scores': exp_train_scores,
            'test_scores': exp_test_scores
        }

    # Plot the final results
    plot_ablation_results(all_results, base_path)

    # Print a summary of test results
    print("\n=== Ablation Study Test Results Summary ===")
    for name, results in all_results.items():
        mean_test_score = np.mean(results['test_scores'])
        std_test_score = np.std(results['test_scores'])
        print(f"  {name}: Mean Test Score = {mean_test_score:.2f} +/- {std_test_score:.2f}")
    
    # Render videos for test seeds across the entire ablation study
    print("\n=== Testing and Rendering ===")
    from render import render
    render(base_path=base_path, seeds_to_run=list(config.seeds.test))

    print(f"\nTotal study time: {(time.time() - t_start) / 60:.2f} minutes.")

def run_single_mode(config):
    """Runs a single experiment, tests it, and renders the result."""
    print(f"=== Starting Single Run: {config.project.run_name} ===")
    t_start = time.time()
    
    base_dir_name = f"{config.project.version}_{config.project.run_name}_{config.mode}"
    base_path = os.path.join("results", base_dir_name)
    os.makedirs(base_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(base_path, "config.yaml"))
    
    # Merge base config with single_run agent config
    full_config = OmegaConf.merge(config, config.single_run)
    
    all_train_scores = []
    trained_model_paths = []
    
    for seed in config.seeds.train:
        print(f"\n--- Training with seed: {seed} ---")
        seed_dir = os.path.join(base_path, f"seed_{seed}")
        models_dir = os.path.join(seed_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        scores, final_model_path = train_single_run(full_config, seed, models_dir)
        all_train_scores.append(scores)
        trained_model_paths.append(final_model_path)
        
    # Plot Training Results
    plt.figure(figsize=(10, 5))
    for i, scores in enumerate(all_train_scores):
        smoothed = smooth_curve(scores, window=10)
        plt.plot(smoothed, label=f'Train Seed {config.seeds.train[i]}')
    plt.title("Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_path, "training_plot.png"))
    plt.close()
    print(f"Training plot saved to {os.path.join(base_path, 'training_plot.png')}")
    
    # Test and Render
    print("\n=== Testing and Rendering ===")
    for i, model_path in enumerate(trained_model_paths):
        train_seed = config.seeds.train[i]
        test_scores = test_single_model(model_path, full_config, config.seeds.test)
        
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        print(f"  Model (Train Seed {train_seed}) Test Score: {mean_score:.2f} +/- {std_score:.2f}")
        
    # Render videos for test seeds across the entire run
    from render import render
    render(base_path=base_path, seeds_to_run=list(config.seeds.test))
        
    print(f"\nTotal run time: {(time.time() - t_start) / 60:.2f} minutes.")

if __name__ == "__main__":
    t_init = time.time()
    main_config = OmegaConf.load("config.yaml")
    if main_config.mode == "single":
        run_single_mode(main_config)
    elif main_config.mode == "ablation":
        run_ablation_study(main_config)
    else:
        print(f"Unknown mode: {main_config.mode}. Please set mode to 'single' or 'ablation'.")
    print(f"\nTotal execution time: {(time.time() - t_init) / 60 / 60:.1f} hours.")
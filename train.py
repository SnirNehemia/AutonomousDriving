import os
import gymnasium as gym
import torch
import highway_env # Import to register highway environments with Gymnasium
import matplotlib.pyplot as plt
import numpy as np
import agents.reinforce
import agents.actor_critic
from omegaconf import OmegaConf
import time


def train_single_run(config, seed, run_name_prefix=""):
    """Trains a single agent configuration for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Setup Directories
    run_id = f"{run_name_prefix}_seed{seed}"
    results_path = os.path.join("results", config.project.ablation_name, run_id)
    os.makedirs(results_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(results_path, "config.yaml"))

    # 2. Create Environment
    env_config_dict = OmegaConf.to_container(config.env_config, resolve=True)
    env = gym.make(config.env.id, config=env_config_dict, render_mode=config.env.render_mode)

    # 3. Create Agent
    # Get state and action dimensions from the environment
    obs_space_shape = env.observation_space.shape
    state_dim = np.prod(obs_space_shape)
    action_dim = env.action_space.n
    
    if config.agent.type == "REINFORCEAgent":
        agent = agents.reinforce.REINFORCEAgent(state_dim, action_dim, hidden_size=config.agent.hidden_size,
                                lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every)
    elif config.agent.type == "A2CAgent":
        agent = agents.actor_critic.A2CAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size,
                                  lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every)
    else:
        raise ValueError(f"Unknown agent type: {config.agent.type}")

    scores = []

    # 4. Training Loop
    for ep in range(config.agent.episodes):
        obs, info = env.reset(seed=seed)
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
        scores.append(ep_reward)

        if ep > 0 and ep % config.agent.save_agent_every == 0:
            agent.save(results_path, f"model_ep{ep}.pth")

    # 5. Save Final Model and Cleanup
    final_model_path = os.path.join(results_path, "model_final.pth")
    agent.save(results_path, "model_final.pth")
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
        agent = agents.actor_critic.A2CAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size)
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
        
        plt.plot(mean_scores, label=name)
        plt.fill_between(range(len(mean_scores)), mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

    plt.title("Ablation Study: Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ablation_path, "ablation_training_plot.png"))
    plt.close()
    print(f"\nAblation plot saved to {os.path.join(ablation_path, 'ablation_training_plot.png')}")


def run_ablation_study():
    """Main function to run the ablation study."""
    t_start = time.time()
    ablation_config = OmegaConf.load("config_ablation.yaml")
    
    ablation_path = os.path.join("results", ablation_config.project.ablation_name)
    os.makedirs(ablation_path, exist_ok=True)
    
    all_results = {}
    
    print(f"=== Starting Ablation Study: {ablation_config.project.ablation_name} ===")
    
    for exp_config in ablation_config.experiments:
        exp_name = exp_config.name
        print(f"\n--- Running Experiment: {exp_name} ---")
        
        # Merge experiment config with base env config
        full_config = OmegaConf.merge(ablation_config, exp_config)
        
        exp_train_scores = []
        trained_model_paths = []
        
        for seed in ablation_config.seeds.train:
            print(f"  Training with seed: {seed}")
            scores, final_model_path = train_single_run(full_config, seed, run_name_prefix=exp_name)
            exp_train_scores.append(scores)
            trained_model_paths.append(final_model_path)
        
        # For simplicity, we test the model from the first training seed
        # A more robust approach could be to test all models and average the results
        model_to_test = trained_model_paths[0]
        exp_test_scores = test_single_model(model_to_test, full_config, ablation_config.seeds.test)
        
        all_results[exp_name] = {
            'train_scores': exp_train_scores,
            'test_scores': exp_test_scores
        }

    # Plot the final results
    plot_ablation_results(all_results, ablation_path)

    # Print a summary of test results
    print("\n=== Ablation Study Test Results Summary ===")
    for name, results in all_results.items():
        mean_test_score = np.mean(results['test_scores'])
        std_test_score = np.std(results['test_scores'])
        print(f"  {name}: Mean Test Score = {mean_test_score:.2f} +/- {std_test_score:.2f}")
    
    print(f"\nTotal study time: {(time.time() - t_start) / 60:.2f} minutes.")


if __name__ == "__main__":
    run_ablation_study()
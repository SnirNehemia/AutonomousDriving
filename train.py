import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import highway_env # Import to register highway environments with Gymnasium
import matplotlib.pyplot as plt
import numpy as np

import agents.reinforce
import agents.actor_critic

from omegaconf import OmegaConf
import time
from render import render

# Load config
config = OmegaConf.load("config.yaml")

def train():
    t_init = time.time()
    # 1. Setup Directories
    run_id = f"{config.project.version}_{config.project.run_name}"
    results_path = f"results/{run_id}"
    os.makedirs(results_path, exist_ok=True)
    OmegaConf.save(config, f"{results_path}/config.yaml") # Save the current config

    # 2. Setup Env with Video Recording
    # Convert OmegaConf to a standard dict for the environment
    env_config_dict = OmegaConf.to_container(config.env_config, resolve=True)
    env = gym.make(
        config.env.id,
        config=env_config_dict,
        render_mode=config.env.render_mode
    )

    # Get state and action dimensions from the environment
    obs_space_shape = env.observation_space.shape
    # The observation is a 2D grid, we flatten it for the agent
    state_dim = np.prod(obs_space_shape) # This is correct for flattened observations
    action_dim = env.action_space.n
    match config.agent.type:
        case "REINFORCEAgent":
            agent = agents.reinforce.REINFORCEAgent(state_dim, action_dim, hidden_size=config.agent.hidden_size,
                                    lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every)
        case "A2CAgent":
            agent = agents.actor_critic.A2CAgent(state_dim, action_dim, actor_hidden_size=config.agent.actor_hidden_size, critic_hidden_size=config.agent.critic_hidden_size,
                                      lr=config.agent.lr, gamma=config.agent.gamma, update_every=config.agent.update_every)

    scores = []
    
    print(f"Starting training {config.agent.type} for {config.agent.episodes} episodes...")
    for ep in range(config.agent.episodes):
        t_episode_start = time.time()
        obs, info = env.reset()
        obs = obs.flatten()
        done = truncated = False
        ep_reward = 0

        while not (done or truncated):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.collect_experience(obs, action, reward, next_obs.flatten(), terminated or truncated)
            obs = next_obs.flatten()
            ep_reward += reward
            done = terminated # Gymnasium split done into terminated/truncated

            agent.try_update(obs, done) # Pass the current obs and done flag for potential mid-episode updates
        scores.append(ep_reward)

        # Save model checkpoint. We save at ep > 0 because at ep=0 the model is untrained.
        if ep > 0 and ep % config.agent.save_agent_every == 0:
            agent.save(results_path, f"model_ep{ep}.pth")

        if ep % 10 == 0:
            print(f"Episode {ep} | Score: {ep_reward:.2f} | time (episode / total): ({(time.time()-t_episode_start):.1f} sec/{(time.time()-t_init)/60:.1f} min)")

    # 3. Save Final Model
    agent.save(results_path, "model.pth")

    # 4. Plot & Save Results
    plt.figure(figsize=(10,5))
    plt.plot(scores, label='Total Reward')
    # Add a moving average to see the trend through the noise
    if len(scores) > 10:
        ma = np.convolve(scores, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(scores)), ma, label='MA (10)')
    
    plt.title(f"Training Progress - {run_id}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"{results_path}/training_score.png")
    plt.close()

    env.close()
    print(f"Training complete. Results saved to {results_path}")

if __name__ == "__main__":
    train()
    render()
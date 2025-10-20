#!/usr/bin/env python3
"""
Evaluation script for trained RL models.

This script provides comprehensive evaluation of trained models on the MuJoCo Hopper
environment, including performance metrics, visualization, and comparison tools.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from environments.hopper_env import make_hopper_env
from environments.wrappers import make_hopper_env_with_wrappers
from algorithms.ppo import PPO
from algorithms.ppo_entropy import PPOEntropy
from training.utils import compare_algorithms, plot_algorithm_comparison


def load_model(algorithm_class, model_path: str, env: gym.Env):
    """Load a trained model."""
    algorithm = algorithm_class(env)
    algorithm.load(model_path)
    return algorithm


def evaluate_model(algorithm, 
                   env: gym.Env, 
                   n_episodes: int = 10, 
                   render: bool = False,
                   deterministic: bool = True) -> dict:
    """Evaluate a model on the environment."""
    print(f"Evaluating model for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    episode_observations = []
    episode_actions = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_obs = []
        episode_act = []
        done = False
        
        while not done:
            action, _ = algorithm.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            episode_obs.append(obs.copy())
            episode_act.append(action.copy())
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_observations.append(np.array(episode_obs))
        episode_actions.append(np.array(episode_act))
        
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")
    
    # Calculate statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_observations": episode_observations,
        "episode_actions": episode_actions
    }
    
    return stats


def plot_evaluation_results(stats: dict, save_path: str = None):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(stats['episode_rewards'], 'b-', alpha=0.7)
    axes[0, 0].axhline(y=stats['mean_reward'], color='r', linestyle='--', 
                      label=f'Mean: {stats["mean_reward"]:.2f}')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(stats['episode_lengths'], 'g-', alpha=0.7)
    axes[0, 1].axhline(y=stats['mean_length'], color='r', linestyle='--',
                      label=f'Mean: {stats["mean_length"]:.2f}')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Reward distribution
    axes[1, 0].hist(stats['episode_rewards'], bins=10, alpha=0.7, color='blue')
    axes[1, 0].axvline(x=stats['mean_reward'], color='r', linestyle='--',
                      label=f'Mean: {stats["mean_reward"]:.2f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Length distribution
    axes[1, 1].hist(stats['episode_lengths'], bins=10, alpha=0.7, color='green')
    axes[1, 1].axvline(x=stats['mean_length'], color='r', linestyle='--',
                      label=f'Mean: {stats["mean_length"]:.2f}')
    axes[1, 1].set_title('Length Distribution')
    axes[1, 1].set_xlabel('Length')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {save_path}")
    else:
        plt.show()


def analyze_policy(algorithm, env: gym.Env, n_samples: int = 1000):
    """Analyze the policy by sampling actions."""
    print(f"Analyzing policy with {n_samples} samples...")
    
    actions = []
    values = []
    
    for _ in range(n_samples):
        obs, _ = env.reset()
        action, value = algorithm.predict(obs, deterministic=False)
        actions.append(action)
        if value is not None:
            values.append(value)
    
    actions = np.array(actions)
    values = np.array(values) if values else None
    
    # Action statistics
    action_stats = {
        "mean": np.mean(actions, axis=0),
        "std": np.std(actions, axis=0),
        "min": np.min(actions, axis=0),
        "max": np.max(actions, axis=0)
    }
    
    # Value statistics
    value_stats = {}
    if values is not None:
        value_stats = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    return {
        "actions": actions,
        "values": values,
        "action_stats": action_stats,
        "value_stats": value_stats
    }


def plot_policy_analysis(analysis: dict, save_path: str = None):
    """Plot policy analysis results."""
    actions = analysis["actions"]
    values = analysis["values"]
    action_stats = analysis["action_stats"]
    value_stats = analysis["value_stats"]
    
    n_actions = actions.shape[1]
    fig, axes = plt.subplots(2, n_actions, figsize=(4 * n_actions, 8))
    
    if n_actions == 1:
        axes = axes.reshape(-1, 1)
    
    # Action distributions
    for i in range(n_actions):
        axes[0, i].hist(actions[:, i], bins=30, alpha=0.7)
        axes[0, i].axvline(x=action_stats["mean"][i], color='r', linestyle='--',
                          label=f'Mean: {action_stats["mean"][i]:.3f}')
        axes[0, i].set_title(f'Action {i} Distribution')
        axes[0, i].set_xlabel('Action Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].legend()
        axes[0, i].grid(True)
    
    # Value distribution (if available)
    if values is not None:
        axes[1, 0].hist(values, bins=30, alpha=0.7, color='green')
        axes[1, 0].axvline(x=value_stats["mean"], color='r', linestyle='--',
                          label=f'Mean: {value_stats["mean"]:.3f}')
        axes[1, 0].set_title('Value Distribution')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Hide other subplots
        for i in range(1, n_actions):
            axes[1, i].set_visible(False)
    else:
        # Hide all value subplots
        for i in range(n_actions):
            axes[1, i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy analysis plots saved to {save_path}")
    else:
        plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained RL models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'ppo_entropy'], 
                       default='ppo', help='Algorithm type')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--analyze_policy', action='store_true',
                       help='Analyze policy behavior')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for policy analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print("Creating evaluation environment...")
    env = make_hopper_env_with_wrappers(
        render_mode="human" if args.render else None,
        normalize_obs=True,
        frame_stack=1,
        reward_scale=1.0,
        max_episode_steps=1000
    )
    
    # Load model
    print(f"Loading {args.algorithm} model from {args.model_path}...")
    if args.algorithm == 'ppo':
        algorithm = load_model(PPO, args.model_path, env)
    elif args.algorithm == 'ppo_entropy':
        algorithm = load_model(PPOEntropy, args.model_path, env)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Evaluate model
    print(f"Evaluating model for {args.n_episodes} episodes...")
    stats = evaluate_model(
        algorithm, 
        env, 
        n_episodes=args.n_episodes,
        render=args.render,
        deterministic=args.deterministic
    )
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min reward: {stats['min_reward']:.2f}")
    print(f"  Max reward: {stats['max_reward']:.2f}")
    print(f"  Mean length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {
            "mean_reward": stats['mean_reward'],
            "std_reward": stats['std_reward'],
            "min_reward": stats['min_reward'],
            "max_reward": stats['max_reward'],
            "mean_length": stats['mean_length'],
            "std_length": stats['std_length'],
            "episode_rewards": stats['episode_rewards'],
            "episode_lengths": stats['episode_lengths']
        }
        json.dump(json_stats, f, indent=2)
    
    # Plot results
    plot_path = output_dir / "evaluation_plots.png"
    plot_evaluation_results(stats, str(plot_path))
    
    # Policy analysis
    if args.analyze_policy:
        print(f"\nAnalyzing policy with {args.n_samples} samples...")
        analysis = analyze_policy(algorithm, env, args.n_samples)
        
        # Print policy statistics
        print(f"Action Statistics:")
        for i, (mean, std) in enumerate(zip(analysis['action_stats']['mean'], 
                                           analysis['action_stats']['std'])):
            print(f"  Action {i}: mean={mean:.3f}, std={std:.3f}")
        
        if analysis['value_stats']:
            print(f"Value Statistics:")
            print(f"  Mean: {analysis['value_stats']['mean']:.3f}")
            print(f"  Std: {analysis['value_stats']['std']:.3f}")
        
        # Plot policy analysis
        policy_plot_path = output_dir / "policy_analysis.png"
        plot_policy_analysis(analysis, str(policy_plot_path))
        
        # Save policy analysis
        policy_analysis_path = output_dir / "policy_analysis.json"
        with open(policy_analysis_path, 'w') as f:
            json.dump({
                "action_stats": {k: v.tolist() for k, v in analysis['action_stats'].items()},
                "value_stats": analysis['value_stats']
            }, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    
    # Clean up
    env.close()


if __name__ == "__main__":
    main()

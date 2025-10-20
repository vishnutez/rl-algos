#!/usr/bin/env python3
"""
Main entry point for the RL algorithms project.

This script provides a simple interface to run training and evaluation
for the MuJoCo Hopper environment using various RL algorithms.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environments.hopper_env import make_hopper_env
from environments.wrappers import make_hopper_env_with_wrappers
from algorithms.ppo import PPO
from algorithms.ppo_entropy import PPOEntropy


def demo_environment():
    """Demonstrate the MuJoCo Hopper environment."""
    print("Creating MuJoCo Hopper environment...")
    
    # Create environment with wrappers
    env = make_hopper_env_with_wrappers(
        render_mode="human",
        normalize_obs=True,
        frame_stack=1,
        reward_scale=1.0,
        max_episode_steps=1000
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few random episodes
    print("\nRunning random episodes...")
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0.0
        step = 0
        
        while step < 100:  # Limit steps for demo
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, steps={step}")
    
    env.close()
    print("Environment demo completed!")


def demo_ppo():
    """Demonstrate PPO algorithm."""
    print("Creating PPO algorithm...")
    
    # Create environment
    env = make_hopper_env_with_wrappers(
        normalize_obs=True,
        frame_stack=1,
        reward_scale=1.0,
        max_episode_steps=1000
    )
    
    # Create PPO algorithm
    algorithm = PPO(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto",
        seed=42,
        verbose=1
    )
    
    print(f"PPO algorithm created with device: {algorithm.device}")
    print("PPO demo completed!")


def demo_ppo_entropy():
    """Demonstrate PPO with entropy weighting."""
    print("Creating PPO with entropy weighting algorithm...")
    
    # Create environment
    env = make_hopper_env_with_wrappers(
        normalize_obs=True,
        frame_stack=1,
        reward_scale=1.0,
        max_episode_steps=1000
    )
    
    # Create PPO with entropy weighting algorithm
    algorithm = PPOEntropy(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_entropy=None,  # Will use -action_dim
        ent_coef_lr=0.001,
        ent_coef_decay=0.999,
        min_ent_coef=0.001,
        max_ent_coef=1.0,
        device="auto",
        seed=42,
        verbose=1
    )
    
    print(f"PPO with entropy weighting algorithm created with device: {algorithm.device}")
    print(f"Target entropy: {algorithm.target_entropy}")
    print("PPO with entropy weighting demo completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='RL Algorithms Demo')
    parser.add_argument('--demo', type=str, 
                       choices=['env', 'ppo', 'ppo_entropy', 'all'],
                       default='all', help='Demo to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL Algorithms for MuJoCo Hopper - Demo")
    print("=" * 60)
    
    if args.demo in ['env', 'all']:
        print("\n1. Environment Demo")
        print("-" * 30)
        demo_environment()
    
    if args.demo in ['ppo', 'all']:
        print("\n2. PPO Algorithm Demo")
        print("-" * 30)
        demo_ppo()
    
    if args.demo in ['ppo_entropy', 'all']:
        print("\n3. PPO with Entropy Weighting Demo")
        print("-" * 30)
        demo_ppo_entropy()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nTo train models, use:")
    print("  python training/train_ppo.py --config configs/ppo_config.yaml")
    print("  python training/train_ppo_entropy.py --config configs/ppo_entropy_config.yaml")
    print("\nTo evaluate models, use:")
    print("  python evaluation/evaluate.py --model_path models/ppo_hopper_final.pkl --algorithm ppo")
    print("  python evaluation/evaluate.py --model_path models/ppo_entropy_hopper_final.pkl --algorithm ppo_entropy")


if __name__ == "__main__":
    main()

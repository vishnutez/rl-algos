#!/usr/bin/env python3
"""
Training script for PPO algorithm on MuJoCo Hopper.

This script provides a complete training pipeline for PPO on the Hopper environment,
including configuration loading, training, evaluation, and model saving.
"""

import argparse
import yaml
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from environments.hopper_env import make_hopper_env
from environments.wrappers import make_hopper_env_with_wrappers
from algorithms.ppo import PPO
from training.utils import TrainingMonitor, EarlyStopping, create_training_callback, save_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, render_mode: str = None) -> gym.Env:
    """Create environment with specified configuration."""
    env_config = config.get('env', {})
    
    # Create base environment
    env = make_hopper_env_with_wrappers(
        render_mode=render_mode,
        normalize_obs=env_config.get('normalize_obs', True),
        frame_stack=env_config.get('frame_stack', 1),
        reward_scale=env_config.get('reward_scale', 1.0),
        max_episode_steps=env_config.get('max_episode_steps', 1000),
        **env_config.get('env_kwargs', {})
    )
    
    return env


def create_algorithm(env: gym.Env, config: dict) -> PPO:
    """Create PPO algorithm with specified configuration."""
    algo_config = config.get('algorithm', {})
    
    algorithm = PPO(
        env=env,
        learning_rate=algo_config.get('learning_rate', 3e-4),
        n_steps=algo_config.get('n_steps', 2048),
        batch_size=algo_config.get('batch_size', 64),
        n_epochs=algo_config.get('n_epochs', 10),
        gamma=algo_config.get('gamma', 0.99),
        gae_lambda=algo_config.get('gae_lambda', 0.95),
        clip_range=algo_config.get('clip_range', 0.2),
        ent_coef=algo_config.get('ent_coef', 0.0),
        vf_coef=algo_config.get('vf_coef', 0.5),
        max_grad_norm=algo_config.get('max_grad_norm', 0.5),
        target_kl=algo_config.get('target_kl', None),
        device=algo_config.get('device', 'auto'),
        seed=algo_config.get('seed', None),
        verbose=algo_config.get('verbose', 1)
    )
    
    return algorithm


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO on MuJoCo Hopper')
    parser.add_argument('--config', type=str, default='configs/ppo_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total number of timesteps to train')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                       help='Number of episodes for evaluation')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Model saving frequency')
    parser.add_argument('--log_dir', type=str, default='logs/ppo',
                       help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='models/ppo',
                       help='Directory to save models')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.seed is not None:
        config['algorithm']['seed'] = args.seed
    if args.verbose is not None:
        config['algorithm']['verbose'] = args.verbose
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create directories
    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    print("Creating training environment...")
    train_env = create_env(config)
    
    print("Creating evaluation environment...")
    eval_env = create_env(config, render_mode="human" if args.render else None)
    
    # Create algorithm
    print("Creating PPO algorithm...")
    algorithm = create_algorithm(train_env, config)
    
    # Create training monitor
    monitor = TrainingMonitor(
        log_dir=str(log_dir),
        save_freq=100,
        plot_freq=1000,
        verbose=args.verbose
    )
    
    # Create early stopping (optional)
    early_stopping = None
    if config.get('training', {}).get('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            monitor=config['training']['early_stopping']['monitor']
        )
    
    # Create training callback
    callback = create_training_callback(monitor, early_stopping)
    
    # Training
    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Algorithm: PPO")
    print(f"Environment: MuJoCo Hopper")
    print(f"Device: {algorithm.device}")
    print(f"Seed: {args.seed}")
    
    try:
        algorithm.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=config.get('training', {}).get('log_interval', 4),
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            tb_log_name="PPO_Hopper",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = model_dir / "ppo_hopper_final.pkl"
        save_model(algorithm, str(final_model_path))
        
        # Final evaluation
        print("\nPerforming final evaluation...")
        final_eval = algorithm.evaluate(
            eval_env, 
            n_eval_episodes=10, 
            deterministic=True, 
            render=args.render
        )
        
        print(f"Final evaluation results:")
        print(f"  Mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"  Mean length: {final_eval['mean_length']:.2f} ± {final_eval['std_length']:.2f}")
        
        # Save training summary
        summary = monitor.get_summary()
        summary_path = log_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining completed successfully!")
        print(f"Logs saved to: {log_dir}")
        print(f"Models saved to: {model_dir}")
        print(f"Best reward achieved: {summary['best_reward']:.2f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save current model
        interrupted_model_path = model_dir / "ppo_hopper_interrupted.pkl"
        save_model(algorithm, str(interrupted_model_path))
        print(f"Model saved to: {interrupted_model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Clean up
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualization tools for RL training and evaluation.

This script provides various visualization tools for analyzing training progress,
comparing algorithms, and understanding agent behavior.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import sys
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def load_training_logs(log_dir: str) -> Dict:
    """Load training logs from directory."""
    log_path = Path(log_dir)
    
    # Find the most recent training stats file
    stats_files = list(log_path.glob("training_stats_*.json"))
    if not stats_files:
        raise FileNotFoundError(f"No training stats found in {log_dir}")
    
    # Load the most recent file
    latest_file = max(stats_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_training_curves(log_dir: str, save_path: str = None):
    """Plot training curves from logs."""
    data = load_training_logs(log_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    if data.get('episode_rewards'):
        rewards = data['episode_rewards']
        axes[0, 0].plot(rewards, alpha=0.7, label='Episode Rewards')
        
        # Add moving average
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                           color='red', linewidth=2, label=f'Moving Avg ({window})')
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Episode lengths
    if data.get('episode_lengths'):
        lengths = data['episode_lengths']
        axes[0, 1].plot(lengths, alpha=0.7, color='green', label='Episode Lengths')
        
        # Add moving average
        if len(lengths) > 10:
            window = min(50, len(lengths) // 10)
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(lengths)), moving_avg, 
                           color='red', linewidth=2, label=f'Moving Avg ({window})')
        
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Training losses
    if data.get('training_losses'):
        losses = data['training_losses']
        for loss_name, loss_values in losses.items():
            if loss_values:
                axes[1, 0].plot(loss_values, label=loss_name)
        
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Evaluation rewards
    if data.get('stats', {}).get('eval_mean_reward'):
        eval_rewards = data['stats']['eval_mean_reward']
        axes[1, 1].plot(eval_rewards, color='purple', label='Eval Mean Reward')
        axes[1, 1].set_title('Evaluation Rewards')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Mean Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def plot_algorithm_comparison(algorithm_dirs: List[str], 
                           algorithm_names: List[str],
                           save_path: str = None):
    """Compare multiple algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithm_dirs)))
    
    for i, (log_dir, name) in enumerate(zip(algorithm_dirs, algorithm_names)):
        try:
            data = load_training_logs(log_dir)
            color = colors[i]
            
            # Episode rewards
            if data.get('episode_rewards'):
                rewards = data['episode_rewards']
                axes[0, 0].plot(rewards, alpha=0.7, color=color, label=name)
                
                # Add moving average
                if len(rewards) > 10:
                    window = min(50, len(rewards) // 10)
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                                   color=color, linewidth=2, alpha=0.8)
            
            # Episode lengths
            if data.get('episode_lengths'):
                lengths = data['episode_lengths']
                axes[0, 1].plot(lengths, alpha=0.7, color=color, label=name)
                
                # Add moving average
                if len(lengths) > 10:
                    window = min(50, len(lengths) // 10)
                    moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
                    axes[0, 1].plot(range(window-1, len(lengths)), moving_avg, 
                                   color=color, linewidth=2, alpha=0.8)
            
            # Training losses
            if data.get('training_losses'):
                losses = data['training_losses']
                for loss_name, loss_values in losses.items():
                    if loss_values:
                        axes[1, 0].plot(loss_values, color=color, label=f"{name} - {loss_name}")
            
            # Evaluation rewards
            if data.get('stats', {}).get('eval_mean_reward'):
                eval_rewards = data['stats']['eval_mean_reward']
                axes[1, 1].plot(eval_rewards, color=color, label=name)
        
        except Exception as e:
            print(f"Error loading data for {name}: {e}")
            continue
    
    # Set titles and labels
    axes[0, 0].set_title('Episode Rewards Comparison')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Episode Lengths Comparison')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Training Losses Comparison')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Evaluation Rewards Comparison')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('Mean Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Algorithm comparison saved to {save_path}")
    else:
        plt.show()


def plot_entropy_adaptation(log_dir: str, save_path: str = None):
    """Plot entropy coefficient adaptation for PPO with entropy weighting."""
    entropy_data_path = Path(log_dir) / "entropy_adaptation_data.json"
    
    if not entropy_data_path.exists():
        print(f"No entropy adaptation data found at {entropy_data_path}")
        return
    
    with open(entropy_data_path, 'r') as f:
        data = json.load(f)
    
    ent_coefs = data['entropy_coefficients']
    entropies = data['entropies']
    target_entropy = data['target_entropy']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Entropy coefficient
    ax1.plot(ent_coefs, label='Entropy Coefficient', color='blue')
    ax1.axhline(y=target_entropy, color='r', linestyle='--', 
               label=f'Target Entropy ({target_entropy:.2f})')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Entropy Coefficient')
    ax1.set_title('Entropy Coefficient Adaptation')
    ax1.legend()
    ax1.grid(True)
    
    # Policy entropy
    ax2.plot(entropies, label='Policy Entropy', color='green')
    ax2.axhline(y=target_entropy, color='r', linestyle='--', 
               label=f'Target Entropy ({target_entropy:.2f})')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Policy Entropy')
    ax2.set_title('Policy Entropy Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Entropy adaptation plot saved to {save_path}")
    else:
        plt.show()


def create_summary_report(log_dirs: List[str], 
                         algorithm_names: List[str],
                         output_path: str):
    """Create a comprehensive summary report."""
    report_data = []
    
    for log_dir, name in zip(log_dirs, algorithm_names):
        try:
            data = load_training_logs(log_dir)
            
            # Extract summary statistics
            episode_rewards = data.get('episode_rewards', [])
            episode_lengths = data.get('episode_lengths', [])
            
            summary = {
                'algorithm': name,
                'total_episodes': len(episode_rewards),
                'best_reward': data.get('best_reward', 0.0),
                'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
                'std_reward': np.std(episode_rewards) if episode_rewards else 0.0,
                'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
                'std_length': np.std(episode_lengths) if episode_lengths else 0.0,
                'training_time': data.get('training_time', 0.0)
            }
            
            report_data.append(summary)
        
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Save report
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = report_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = report_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Print summary
    print("\nTraining Summary Report:")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nReport saved to: {csv_path} and {json_path}")
    
    return df


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize RL training results')
    parser.add_argument('--log_dirs', nargs='+', required=True,
                       help='Directories containing training logs')
    parser.add_argument('--algorithm_names', nargs='+', required=True,
                       help='Names of algorithms for comparison')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--plot_type', type=str, 
                       choices=['training_curves', 'comparison', 'entropy', 'all'],
                       default='all', help='Type of plot to generate')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot_type in ['training_curves', 'all']:
        # Plot training curves for each algorithm
        for log_dir, name in zip(args.log_dirs, args.algorithm_names):
            print(f"Plotting training curves for {name}...")
            save_path = output_dir / f"{name}_training_curves.png" if args.save_plots else None
            plot_training_curves(log_dir, str(save_path) if save_path else None)
    
    if args.plot_type in ['comparison', 'all']:
        # Plot algorithm comparison
        print("Plotting algorithm comparison...")
        save_path = output_dir / "algorithm_comparison.png" if args.save_plots else None
        plot_algorithm_comparison(args.log_dirs, args.algorithm_names, 
                                str(save_path) if save_path else None)
    
    if args.plot_type in ['entropy', 'all']:
        # Plot entropy adaptation (if available)
        for log_dir, name in zip(args.log_dirs, args.algorithm_names):
            if 'entropy' in name.lower():
                print(f"Plotting entropy adaptation for {name}...")
                save_path = output_dir / f"{name}_entropy_adaptation.png" if args.save_plots else None
                plot_entropy_adaptation(log_dir, str(save_path) if save_path else None)
    
    # Create summary report
    print("Creating summary report...")
    report_path = output_dir / "training_summary"
    create_summary_report(args.log_dirs, args.algorithm_names, str(report_path))
    
    print(f"\nVisualization completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

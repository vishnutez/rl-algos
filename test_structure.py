#!/usr/bin/env python3
"""
Test script to verify the project structure without requiring dependencies.
"""

import os
from pathlib import Path

def test_structure():
    """Test the project structure."""
    print("Testing RL Algorithms Project Structure")
    print("=" * 50)
    
    # Check main directories
    directories = [
        "environments",
        "algorithms", 
        "training",
        "evaluation",
        "configs",
        "results"
    ]
    
    print("\n1. Checking directories:")
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (missing)")
    
    # Check key files
    key_files = [
        "README.md",
        "environment.yaml",
        "requirements.txt",
        "main.py",
        "environments/__init__.py",
        "environments/hopper_env.py",
        "environments/wrappers.py",
        "algorithms/__init__.py",
        "algorithms/base.py",
        "algorithms/ppo.py",
        "algorithms/ppo_entropy.py",
        "training/__init__.py",
        "training/utils.py",
        "training/train_ppo.py",
        "training/train_ppo_entropy.py",
        "evaluation/__init__.py",
        "evaluation/evaluate.py",
        "evaluation/visualize.py",
        "configs/ppo_config.yaml",
        "configs/ppo_entropy_config.yaml"
    ]
    
    print("\n2. Checking key files:")
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
    
    # Check file sizes
    print("\n3. File sizes:")
    for file_path in key_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file_path}: {size:,} bytes")
    
    print("\n4. Project structure summary:")
    print("  - MuJoCo Hopper environment implementation")
    print("  - PPO algorithm with clean implementation")
    print("  - PPO with entropy weighting")
    print("  - Training scripts with configuration support")
    print("  - Evaluation and visualization tools")
    print("  - Comprehensive documentation")
    
    print("\n" + "=" * 50)
    print("Project structure test completed!")

if __name__ == "__main__":
    test_structure()

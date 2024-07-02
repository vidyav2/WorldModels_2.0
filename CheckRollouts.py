import numpy as np
import gymnasium as gym
import os
import random

# Load the provided NPZ files
"""def load_rollout_data(npz_file):
    data = np.load(npz_file)
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['terminals']
    return observations, actions, rewards, dones

# Function to check the correctness of the rollouts
def check_rollout_correctness(npz_file):
    observations, actions, rewards, dones = load_rollout_data(npz_file)
    
    # Print debug information
    print(f"File: {npz_file}")
    print(f"Length of observations: {len(observations)}")
    print(f"Length of actions: {len(actions)}")
    print(f"Length of rewards: {len(rewards)}")
    print(f"Length of dones: {len(dones)}")
    print(f"Sample dones: {dones[:1000]}")

    incorrect_frames = []
    
    for i in range(len(dones) - 1):
        if dones[i] and i < 998:
            incorrect_frames.append(i)
    
    if len(dones) == 1000 and not dones[999]:
        incorrect_frames.append(999)
    
    if not any(dones):
        incorrect_frames.append("No termination detected in the entire episode")
    
    # Additional detailed checks
    for i in range(len(observations) - 1):
        if dones[i] and dones[i + 1] == False:
            incorrect_frames.append(f"Abrupt termination at frame {i}")
    
    return incorrect_frames

# List all npz files in the provided directory
def check_random_rollouts(directory, num_files=5):
    npz_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(subdir, file))
    
    # Select a random subset of npz files
    random_files = random.sample(npz_files, min(num_files, len(npz_files)))
    issues = {}
    
    for npz_file in random_files:
        print(f"Checking file: {npz_file}")
        incorrect_frames = check_rollout_correctness(npz_file)
        if incorrect_frames:
            issues[npz_file] = incorrect_frames
    
    return issues

# Directory containing the rollout files
rollout_directory = "/mnt/c/Users/Vidyavarshini/WorldModels_2.0/datasets/carracing"

# Check random rollouts in the provided directory
rollout_issues = check_random_rollouts(rollout_directory)
print(rollout_issues)"""



env=gym.make("CarRacing-v2")
print('Action Space:', env.action_space)
print('Observation Space:', env.observation_space)
print('Reward Range:', env.reward_range)
print('Reward Range:', env.termination)
print(env.__doc__)




















"""import numpy as np

# Load the provided NPZ file and print its keys
def inspect_npz_file(npz_file):
    data = np.load(npz_file)
    print(f"Keys in {npz_file}: {list(data.keys())}")

# Directory containing the rollout files
npz_file_path = "/mnt/c/Users/Vidyavarshini/WorldModels_2.0/datasets/carracing/thread_0/rollout_0.npz"

# Inspect the NPZ file
inspect_npz_file(npz_file_path)"""

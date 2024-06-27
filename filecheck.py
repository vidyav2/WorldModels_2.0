"""This is a test file to check if the data is being loaded correctly
import numpy as np
import os

# Set the path to the directory containing your .npz files
data_dir = "datasets/carracing/thread_1/"

# List all .npz files in the directory
npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# Loop through the .npz files and inspect their contents
for file_name in npz_files:
    file_path = os.path.join(data_dir, file_name)
    with np.load(file_path) as data:
        observations = data['observations']  # Assuming observations are stored as 'observations'
        actions = data['actions']  # Assuming actions are stored as 'actions'
        rewards = data['rewards']  # Assuming rewards are stored as 'rewards'
        terminals = data['terminals']  # Assuming terminals are stored as 'terminals'

    # Print some information about the loaded data
    print(f"File: {file_name}")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Terminals shape: {terminals.shape}")
    print(f"Number of frames: {len(observations)}")
    print("\n")"""


"""import os

# Set the path to the directory containing your .npz files
data_dir = "datasets/carracing"

# List all .npz files in the directory
npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# Print the list of .npz files found
print("Found {} files".format(len(npz_files)))
print("File list:", npz_files)"""



import numpy as np
import gym
import matplotlib.pyplot as plt

def visualize_rollout(data_path):
    data = np.load(data_path)
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    terminals = data['terminals']

    env = gym.make("CarRacing-v2", render_mode="human")
    env.reset()

    for i, obs in enumerate(observations):
        env.render()
        action = actions[i]
        env.step(action)

        if terminals[i]:
            print(f"Rollout ended at frame {i} with reward {rewards[i]}")
            break

    env.close()

# Example usage
visualize_rollout("datasets/carracing/thread_3/rollout_2.npz")

"""import numpy as np
import os

def check_statistics(data_dir, num_rollouts):
    total_rewards = []
    episode_lengths = []

    for i in range(num_rollouts):
        data_path = os.path.join(data_dir, f'rollout_{i}.npz')
        data = np.load(data_path)
        rewards = data['rewards']
        terminals = data['terminals']

        total_rewards.append(np.sum(rewards))
        episode_lengths.append(len(terminals))

    print("Average total reward:", np.mean(total_rewards))
    print("Average episode length:", np.mean(episode_lengths))

# Example usage
check_statistics("datasets/carracing/thread_4/", 50)"""


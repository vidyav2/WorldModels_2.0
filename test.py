"""import cv2

import numpy as np

a = np.load("/datasets/carracing/12332333312.npz")["obs"]

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('output.avi', fourcc, 50.0, (84, 96))

for i in range(a.shape[0]):
    out.write(a[i])

out.release()"""


"""import numpy as np
import os

# Define the dataset path
dataset_path = 'datasets/carracing'
# List the thread directories
thread_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if 'thread' in d and os.path.isdir(os.path.join(dataset_path, d))]

# Check the contents of the first file in each thread directory
for dir in thread_dirs:
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npz')]
    if files:
        file_path = files[0]
        data = np.load(file_path)
        print(f"Contents of {file_path}: {data.files}")
        # Displaying the contents of the file
        for key in data.files:
            print(f"{key}: {data[key]}")
            print(f"{key} shape: {data[key].shape}")
    else:
        print(f"No .npz files found in the directory {dir}.")"""


"""import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('datasets/carracing/thread_5/rollout_0.npz')

# Extract terminals
terminals = data['terminals']

# Print number of terminal states and their positions
num_terminals = np.sum(terminals)
terminal_positions = np.where(terminals == True)
print(f"Number of terminal states: {num_terminals}")
print(f"Terminal state positions: {terminal_positions}")

# Visualize terminal states
plt.plot(terminals)
plt.title('Terminal States')
plt.xlabel('Step')
plt.ylabel('Is Terminal')
plt.show()

# Check episode lengths
episode_lengths = np.diff(np.where(terminals == True)[0])
print(f"Episode lengths: {episode_lengths}")"""



"""import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('datasets/carracing/thread_7/rollout_0.npz')

# Extract terminals
terminals = data['terminals']

# Print number of terminal states and their positions
num_terminals = np.sum(terminals)
terminal_positions = np.where(terminals == True)
print(f"Number of terminal states: {num_terminals}")
print(f"Terminal state positions: {terminal_positions}")

# Visualize terminal states
plt.plot(terminals)
plt.title('Terminal States')
plt.xlabel('Step')
plt.ylabel('Is Terminal')
plt.savefig('terminal_states.png')  # Save the plot as a PNG file
print("Saved the terminal states plot as terminal_states.png")

# Check episode lengths
episode_lengths = np.diff(np.where(terminals == True)[0])
print(f"Episode lengths: {episode_lengths}")"""



import numpy as np
import matplotlib.pyplot as plt
import os
import random

def load_rollout(filepath):
    data = np.load(filepath)
    observations = data['observations']
    rewards = data['rewards']
    dones = data['dones']
    return observations, rewards, dones

def visualize_rollout(observations, rewards, dones, sample_frames=10, save_dir='rollout_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_frames = len(observations)
    frames_to_show = list(range(min(sample_frames, num_frames))) + list(range(max(0, num_frames - sample_frames), num_frames))
    
    for i in frames_to_show:
        plt.imshow(observations[i])
        plt.title(f"Frame {i}, Reward: {rewards[i]}, Done: {dones[i]}")
        plt.savefig(os.path.join(save_dir, f"frame_{i}.png"))
        plt.close()

def check_rollout(filepath, save_dir='rollout_images'):
    observations, rewards, dones = load_rollout(filepath)

    # Check lengths
    assert len(observations) == len(rewards) == len(dones), "Length mismatch in rollout arrays."
    
    # Check terminal states
    if not any(dones):
        print(f"No terminal state found in rollout: {filepath}")
    else:
        print(f"Terminal state found in rollout: {filepath}")

    print(f"Rollout length: {len(observations)} frames")
    visualize_rollout(observations, rewards, dones, save_dir=save_dir)

def validate_rollouts(rollout_dir, sample_size=5):
    rollout_files = [os.path.join(rollout_dir, f) for f in os.listdir(rollout_dir) if f.endswith('.npz')]
    assert len(rollout_files) > 0, "No rollout files found."

    # Randomly sample rollouts
    sample_files = random.sample(rollout_files, min(sample_size, len(rollout_files)))

    for idx, rollout_file in enumerate(sample_files):
        print(f"Validating rollout {idx + 1}/{len(sample_files)}: {rollout_file}")
        check_rollout(rollout_file, save_dir=f'rollout_images_{idx}')

def main():
    rollout_dir = 'datasets/carracing/thread_0/'
    validate_rollouts(rollout_dir, sample_size=5)

if __name__ == "__main__":
    main()


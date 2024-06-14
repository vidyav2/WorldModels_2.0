"""This is a test file to check if the data is being loaded correctly"""
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
    print("\n")


"""import os

# Set the path to the directory containing your .npz files
data_dir = "datasets/carracing"

# List all .npz files in the directory
npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# Print the list of .npz files found
print("Found {} files".format(len(npz_files)))
print("File list:", npz_files)"""

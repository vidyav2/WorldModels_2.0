import numpy as np

# Path to one of the .npz files
file_path = 'datasets/carracing/thread_0/rollout_0.npz'

# Load the .npz file and print its keys
with np.load(file_path) as data:
    print("Keys in the .npz file:", data.keys())

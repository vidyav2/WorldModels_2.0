import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os

def check_rollouts(data_dir, thread_count, rollouts_per_thread):
    for thread_id in range(thread_count):
        thread_dir = join(data_dir, f'thread_{thread_id}')
        for rollout_id in range(rollouts_per_thread):
            filepath = join(thread_dir, f'rollout_{rollout_id}.npz')
            if os.path.exists(filepath):
                data = np.load(filepath)
                rewards = data['rewards']
                terminals = data['terminals']
                
                # Plot rewards
                plt.figure()
                plt.plot(rewards)
                plt.title(f'Rewards for rollout_{rollout_id}.npz in thread_{thread_id}')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.show()
                
                # Print terminal states
                print(f'Terminal states for rollout_{rollout_id}.npz in thread_{thread_id}: {terminals}')
            else:
                print(f'File not found: {filepath}')

check_rollouts('datasets/carracing', 8, 2)

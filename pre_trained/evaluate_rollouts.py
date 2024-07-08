import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_rollout(file_path, output_dir):
    data = np.load(file_path)
    observations = data['observations']
    rewards = data['rewards']
    actions = data['actions']
    terminals = data['terminals']

    print(f"Total frames: {len(observations)}")
    print(f"Cumulative reward: {np.sum(rewards)}")
    print(f"Actions shape: {actions.shape}")
    print(f"Terminals shape: {terminals.shape}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save some frames and corresponding actions
    for i in range(0, len(observations), max(1, len(observations)//10)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].imshow(observations[i])
        axs[0].set_title(f"Frame {i}")

        axs[1].bar(['Steering', 'Gas', 'Brake'], actions[i])
        axs[1].set_title(f"Actions at Frame {i}")

        # Save the figure
        fig_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_frame_{i}.png")
        plt.savefig(fig_path)
        plt.close(fig)

def evaluate_rollouts(rollout_dir, output_dir):
    cumulative_rewards = []
    for filename in os.listdir(rollout_dir):
        if filename.endswith('.npz'):
            file_path = os.path.join(rollout_dir, filename)
            data = np.load(file_path)
            rewards = data['rewards']
            cumulative_rewards.append(np.sum(rewards))
            print(f"{filename} - Cumulative Reward: {np.sum(rewards)}")

            # Visualize and save images
            visualize_rollout(file_path, output_dir)
    return cumulative_rewards

def check_track_completion(rollout_dir):
    track_completions = []
    for filename in os.listdir(rollout_dir):
        if filename.endswith('.npz'):
            file_path = os.path.join(rollout_dir, filename)
            data = np.load(file_path)
            terminals = data['terminals']
            completed = not any(terminals)  # Check if the track was completed
            track_completions.append(completed)
            print(f"{filename} - Completed Track: {completed}")
    return track_completions

# Run the analysis
rollout_dir = 'rollouts'
output_dir = 'visualizations'

print("Visualizing rollouts...")
for filename in os.listdir(rollout_dir):
    if filename.endswith('.npz'):
        print(f"Visualizing {filename}")
        visualize_rollout(os.path.join(rollout_dir, filename), output_dir)

print("\nEvaluating cumulative rewards...")
cumulative_rewards = evaluate_rollouts(rollout_dir, output_dir)
print(f"Average Cumulative Reward: {np.mean(cumulative_rewards)}")
print(f"Max Cumulative Reward: {np.max(cumulative_rewards)}")
print(f"Min Cumulative Reward: {np.min(cumulative_rewards)}")

print("\nChecking track completions...")
track_completions = check_track_completion(rollout_dir)
print(f"Tracks Completed: {sum(track_completions)} out of {len(track_completions)}")

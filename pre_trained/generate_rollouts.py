import argparse
from os.path import join, exists
import os
import numpy as np
import gymnasium as gym
import torch
from pretrained_model import load_pretrained_agent

def preprocess_observation(observation):
    # Normalize observation to the expected input format
    observation_resized = observation / 255.0  # Normalize pixel values to [0, 1]
    return observation_resized

def generate_data(rollouts, data_dir, model):
    assert exists(data_dir), "The data directory does not exist..."
    env = gym.make("CarRacing-v2")
    seq_len = 1000

    for i in range(rollouts):
        observation, _ = env.reset()
        s_rollout = []
        r_rollout = []
        a_rollout = []
        d_rollout = []

        t = 0
        while True:
            obs_processed = preprocess_observation(observation)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).permute(2, 0, 1)  # Rearrange to (C, H, W)
            action = model.step(obs_tensor)[0]
            observation, reward, terminated, truncated, _ = env.step(action)
            
            s_rollout.append(observation)
            r_rollout.append(reward)
            a_rollout.append(action)
            d_rollout.append(terminated)
            
            t += 1
            if terminated or t >= seq_len:
                print(f"> End of rollout {i}, {len(s_rollout)} frames...")
                np.savez(join(data_dir, f'rollout_{i}'),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    args = parser.parse_args()

    pretrained_model_path = 'original.npz'  # Use the new .npz file
    model = load_pretrained_agent(pretrained_model_path)
    
    data_dir = 'rollouts/'
    if not exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"Generating {args.rollouts} rollouts in directory {data_dir}...")
    generate_data(args.rollouts, data_dir, model)
    print("Rollout generation completed.")

import argparse
import os
from os.path import join, exists
import numpy as np
import gymnasium as gym
import torch
from concurrent.futures import ThreadPoolExecutor
import pygame
from pretrained_model import load_pretrained_agent

def preprocess_observation(observation):
    observation_resized = observation / 255.0  # Normalize pixel values to [0, 1]
    return observation_resized

def generate_single_rollout(env, model, seq_len, rollout_index, data_dir):
    try:
        env.reset()
        # Initialize car wheels properly
        env.render()
        observation, _ = env.reset()
        s_rollout = []
        r_rollout = []
        a_rollout = []
        d_rollout = []

        t = 0
        while True:
            obs_processed = preprocess_observation(observation)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).permute(2, 0, 1).cuda()  # Rearrange to (C, H, W) and move to GPU
            action = model.step(obs_tensor)[0]
            observation, reward, terminated, truncated, _ = env.step(action)
            
            s_rollout.append(observation)
            r_rollout.append(reward)
            a_rollout.append(action)
            d_rollout.append(terminated)
            
            t += 1
            if terminated or t >= seq_len:
                np.savez(join(data_dir, f'rollout_{rollout_index}'),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break
    except pygame.error as e:
        print(f"Pygame error: {e}")
    finally:
        env.close()

def generate_data(rollouts, data_dir, model, threads=4):
    assert exists(data_dir), "The data directory does not exist..."
    seq_len = 1000

    envs = [gym.make("CarRacing-v2") for _ in range(threads)]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(generate_single_rollout, envs[i % threads], model, seq_len, i, data_dir)
                   for i in range(rollouts)]
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, required=True, help="Where to place rollouts")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads for rollout generation")
    args = parser.parse_args()

    pretrained_model_path = 'pre_trained/original.npz'  # Use the new .npz file
    model = load_pretrained_agent(pretrained_model_path).cuda()  # Move model to GPU

    print(f"Generating {args.rollouts} rollouts in directory {args.dir} with {args.threads} threads...")
    generate_data(args.rollouts, args.dir, model, args.threads)
    print("Rollout generation completed.")

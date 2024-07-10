import argparse
from os.path import join, exists
import numpy as np
import gymnasium as gym
import torch
import concurrent.futures
import pygame
from pretrained_model import load_pretrained_agent

def preprocess_observation(observation):
    observation_resized = observation / 255.0  # Normalize pixel values to [0, 1]
    return observation_resized

def generate_single_rollout(rollout_id, data_dir, model, device):
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    seq_len = 1000
    observation, _ = env.reset()
    s_rollout = []
    r_rollout = []
    a_rollout = []
    d_rollout = []

    t = 0
    while True:
        try:
            obs_processed = preprocess_observation(observation)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).permute(2, 0, 1).to(device)
            action = model.step(obs_tensor)[0]
            observation, reward, terminated, truncated, _ = env.step(action)
            
            s_rollout.append(observation)
            r_rollout.append(reward)
            a_rollout.append(action)
            d_rollout.append(terminated)
            
            t += 1
            if terminated or t >= seq_len:
                print(f"> End of rollout {rollout_id}, {len(s_rollout)} frames...")
                np.savez(join(data_dir, f'rollout_{rollout_id}'),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break
        except pygame.error as e:
            print(f"Pygame error: {e}")
            env.close()
            env = gym.make("CarRacing-v2", render_mode="rgb_array")
            observation, _ = env.reset()

def generate_data(num_rollouts, data_dir, model, num_threads, device):
    assert exists(data_dir), "The data directory does not exist..."
    model.set_device(device)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(generate_single_rollout, i, data_dir, model, device) for i in range(num_rollouts)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads to use")
    parser.add_argument('--use-gpu', action='store_true', help="Use GPU if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    pretrained_model_path = 'pre_trained/original.npz'
    model = load_pretrained_agent(pretrained_model_path)
    
    print(f"Generating {args.rollouts} rollouts in directory {args.dir} with {args.threads} threads...")
    generate_data(args.rollouts, args.dir, model, args.threads, device)
    print("Rollout generation completed.")

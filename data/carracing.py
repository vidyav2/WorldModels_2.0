import argparse
from os.path import join, exists
import gymnasium as gym
import numpy as np
from utils.misc import sample_mixed_policy
from PIL import Image

def generate_data(rollouts, data_dir, noise_type, seq_len=1000, gif_file=None):
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make("CarRacing-v2", render_mode='rgb_array')
    all_frames = []

    for i in range(rollouts):
        obs, _ = env.reset()
        if gif_file:
            frame = env.render()
            frames = [Image.fromarray(frame).resize((320, 240))]

        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type in ['brown', 'mixed']:
            a_rollout = sample_mixed_policy(env.action_space, seq_len, env)

        s_rollout, r_rollout, d_rollout = [], [], []

        for t in range(seq_len):
            action = a_rollout[t]
            s, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if gif_file:
                frame = env.render()
                frames.append(Image.fromarray(frame).resize((320, 240)))
            
            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(done)
            
            if done or t == seq_len - 1:
                print(f"> End of rollout {i}, {t+1} frames...")
                break

        np.savez(join(data_dir, f'rollout_{i}'),
                 observations=np.array(s_rollout),
                 rewards=np.array(r_rollout),
                 actions=np.array(a_rollout),
                 terminals=np.array(d_rollout))
        if gif_file:
            all_frames.extend(frames)

    if gif_file:
        all_frames[0].save(gif_file, save_all=True, append_images=all_frames[1:], loop=0, duration=40)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown', 'mixed'], help='Noise type used for action sampling.', default='mixed')
    parser.add_argument('--gif', type=str, help="Output GIF file", default=None)
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy, gif_file=args.gif)

import argparse
from os.path import join, exists
import gym
import numpy as np
from utils.misc import sample_continuous_policy

def generate_data(rollouts, data_dir, noise_type):
    """Generates data"""
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make("CarRacing-v2", render_mode="human")
    seq_len = 1000

    for i in range(rollouts):
        obs, _ = env.reset()
        if hasattr(env, 'viewer') and env.viewer:
            env.viewer.window.dispatch_events()

        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        cumulative_reward = 0
        while t < seq_len:
            action = a_rollout[t]
            action += np.random.normal(0, 0.1, size=action.shape)  # Adding noise
            action = np.clip(action, -1, 1)  # Clipping actions to valid range
            t += 1

            step_output = env.step(action)
            if len(step_output) == 5:
                s, r, done, truncated, _ = step_output
                done = done or truncated  # Handle truncated as done
            else:
                s, r, done, _ = step_output

            if hasattr(env, 'viewer') and env.viewer:
                env.viewer.window.dispatch_events()
            env.render()

            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(bool(done))
            cumulative_reward += r
            print(f"Step {t}, Reward: {r}, Done: {done}")  # Detailed step reward
            if done:
                print(f"> End of rollout {i}, {len(s_rollout)} frames, final reward: {cumulative_reward}")
                break

        np.savez(join(data_dir, f'rollout_{i}'),
                 observations=np.array(s_rollout),
                 rewards=np.array(r_rollout),
                 actions=np.array(a_rollout[:t]),
                 terminals=np.array(d_rollout))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)

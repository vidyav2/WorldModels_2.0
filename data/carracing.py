"""
Generating data from the CarRacing gym environment.
"""
import argparse
from os.path import join, exists
import gym
import numpy as np
from utils.misc import sample_continuous_policy


def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """
    Generates data for the CarRacing environment.

    Args:
        rollouts: Number of rollouts to generate data for.
        data_dir: Directory to save the generated data.
        noise_type: Type of noise to use for action sampling ('white' or 'brown').
    """
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make("CarRacing-v2")
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        # Render the environment during data generation
        env.render()

        # Choose noise type for action sampling
        if noise_type == "white":
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == "brown":
            # Ensure proper import and usage of sample_continuous_policy
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1.0 / 50)
        else:
            raise ValueError("Invalid noise type: {}".format(noise_type))

        s_rollout = []
        r_rollout = []
        d_rollout = []

        # Handle potential exceeding of rollout length
        t = 0
        while t < len(a_rollout):
            action = a_rollout[t]
            t += 1

            step_result = env.step(action)
            s, r, done, _, _ = step_result

            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(done)

            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                break

        # Save data using np.savez with proper format and extension
        np.savez(join(data_dir, "rollout_{}.npz".format(i)),
                 observations=np.array(s_rollout),
                 rewards=np.array(r_rollout),
                 actions=np.array(a_rollout),
                 terminals=np.array(d_rollout))

    print("Data generation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)

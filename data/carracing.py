import argparse
from os.path import join, exists
import gym
import numpy as np
from utils.misc import sample_continuous_policy

def generate_data(rollouts, data_dir, noise_type):
    """
    Generates data for the CarRacing environment.

    Args:
        rollouts: Number of rollouts to generate data for.
        data_dir: Directory to save the generated data.
        noise_type: Type of noise to use for action sampling ('white' or 'brown').
    """
    assert exists(data_dir), "The data directory does not exist..."
    print(f"Generating {rollouts} rollouts with {noise_type} noise...")

    # Initialize the CarRacing environment
    env = gym.make("CarRacing-v2")
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        env.render()  # Render the environment during data generation
        #print(f"Starting rollout {i+1}/{rollouts}...")

        # Choose noise type for action sampling
        if noise_type == "white":
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == "brown":
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1.0 / 50)
        else:
            raise ValueError("Invalid noise type: {}".format(noise_type))

        s_rollout = []
        r_rollout = []
        d_rollout = []

        # Generate rollout data
        t = 0
        while t < len(a_rollout):
            action = a_rollout[t]
            t += 1

            # Step the environment and capture the results
            step_result = env.step(action)
            if len(step_result) == 4:
                s, r, done, _ = step_result
            elif len(step_result) == 5:
                s, r, done, _, _ = step_result

            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(done)

            if done:
                print(f"End of rollout {i+1}, {len(s_rollout)} frames.")
                break

        # Save the rollout data
        np.savez(join(data_dir, f"rollout_{i}.npz"),
                 observations=np.array(s_rollout),
                 rewards=np.array(r_rollout),
                 actions=np.array(a_rollout),
                 terminals=np.array(d_rollout))
        #print(f"Saved rollout {i+1} to {join(data_dir, f'rollout_{i}.npz')}")

    print(f"Completed generating {rollouts} rollouts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, required=True, help="Number of rollouts")
    parser.add_argument('--dir', type=str, required=True, help="Directory to save rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.', default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)

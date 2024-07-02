import argparse
from os.path import join, exists
import gymnasium as gym
import numpy as np
from utils.misc import sample_continuous_policy
import imageio

def generate_data(rollouts, data_dir, noise_type):
    assert exists(data_dir), "The data directory does not exist..."

    #env = gym.make("CarRacing-v2",continuous=False)
    env = gym.make("CarRacing-v2", render_mode="rgb_array", lap_complete_percent=1)
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
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
        #cumulative_reward = 0
        #done = False
        while True:
            print("tester",t)
            #print("action",{a_rollout[t-1]})
            action = a_rollout[t]
            t += 1
            #action += np.random.normal(0, 0.05, size=action.shape)  # Reduced noise
            #action = np.clip(action, -1, 1)  # Clipping actions to valid range

            #s,r,done, _, _= env.step(action)
            s, r, done,_,_ = env.step(action)
            #done = terminated or truncated

            if hasattr(env, 'viewer') and env.viewer:
                env.viewer.window.dispatch_events()
            env.render()

            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(done)


            print(f"Reward: {r}, Done: {done}, Action: {action}")  # Detailed step reward

            if done:
                print(f"> End of rollout {i}, {len(s_rollout)} frames")
                #break

            #t += 1

            #if  t < seq_len:
            #s_rollout.append(np.zeros_like(s))
            #r_rollout.append(0)
            #d_rollout.append(True)
            #t += 1

                np.savez(join(data_dir, f'rollout_{i}'),
                    observations=np.array(s_rollout),
                    rewards=np.array(r_rollout),
                    actions=np.array(a_rollout),
                    terminals=np.array(d_rollout))
                break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)

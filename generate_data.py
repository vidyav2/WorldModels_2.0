import gym
import numpy as np

import matplotlib.pyplot as plt

from utils.misc import sample_continuous_policy

import multiprocessing

gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 128, 128

def generate_data():
    NUM_EPISODES = 40
    NUM_STEPS = 3000

    SKIP_STEPS = 3

    for episode in range(NUM_EPISODES):
        env = gym.make("CarRacing-v2", render_mode="rgb_array")

        obs, r, done = [], [], []
        actions = []

        env.reset()

        a_rollouts = sample_continuous_policy(env.action_space, NUM_STEPS, 1. / 50)

        for step in range(NUM_STEPS):
            action = a_rollouts[step]
            
            observation, reward, terminated, truncated, info = env.step(action)

            obs.append(observation[0:112,:,:])
            actions.append(action)
            r.append(reward)

            if terminated:
                done.append(1)
                break
            
            done.append(0)

        env.close()

        # create random filename
        filename = "datasets/carracing/" + str(np.random.randint(100000000000)) + ".npz"
        #filename = "datasets/carracing/thread_0" + ".npz"

        obs = np.array(obs)
        actions = np.array(actions)
        r = np.array(r)
        done = np.array(done)

        # save data
        np.savez(filename, obs=obs, action=actions, r=r, done=done)

generate_data()
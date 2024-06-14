import gym
import numpy as np

import matplotlib.pyplot as plt

from utils.misc import sample_continuous_policy

import multiprocessing
import cv2

from models.vae import VAE

import torch

gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 96, 96

def generate_data():
    NUM_EPISODES = 10
    NUM_STEPS = 1000

    SKIP_STEPS = 3

    vae = VAE(3, 32)

    count = 0

    vae.load_state_dict(torch.load("logs/vae/best.tar")["state_dict"])

    for episode in range(NUM_EPISODES):
        env = gym.make("CarRacing-v2", render_mode="rgb_array")

        obs, r, done = [], [], []
        actions = []

        env.reset()

        for step in range(25):
            action = np.array([0,0,0])
            
            observation, reward, terminated, truncated, info = env.step(action)

        a_rollouts = sample_continuous_policy(env.action_space, NUM_STEPS, 1. / 50)

        for step in range(NUM_STEPS):
            action = a_rollouts[step]
            
            observation, reward, terminated, truncated, info = env.step(action)

            obs = observation[0:84,:,:]

            # resize to 64x64 
            obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_LINEAR)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

            # save data
            cv2.imwrite("fid/real/" + str(count) + ".png", obs)

            z = torch.randn(1, 32)

            obs = vae.decoder(z)
            # permute to (H, W, C)
            obs = obs.permute(0, 2, 3, 1)
            # convert to numpy
            obs = obs.detach().numpy()[0]

            # change valies to 0-255
            obs = (obs * 255).astype(np.uint8)
            print(f"Counter: {count}    ", end="\r")
            # save data
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imwrite("fid/fake/" + str(count) + ".png", obs)

            # if terminated:
            #     done.append(1)
            #     break
            
            done.append(0)

            count += 1

        env.close()    

generate_data()
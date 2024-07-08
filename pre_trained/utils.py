# utils.py
import torch

def preprocess_observation(observation):
    """ Preprocess observation for the model """
    return observation.flatten()

def predict_action(observation, model):
    observation = torch.tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        action = model(observation).numpy()
    return action

# controller.py
import torch
import torch.nn as nn
from noisy_layer import NoisyLinear  # Import NoisyLinear

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = NoisyLinear(latents + recurrents, actions)  # Use NoisyLinear

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)

    def reset_noise(self):  # Add this method
        self.fc.reset_noise()

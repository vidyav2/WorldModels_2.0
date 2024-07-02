import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.050):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
    
    def print_parameters(self):
        print("Weight Mu: \n", self.weight_mu)
        print("Weight Sigma: \n", self.weight_sigma)
        print("Weight Epsilon: \n", self.weight_epsilon)
        print("Bias Mu: \n", self.bias_mu)
        print("Bias Sigma: \n", self.bias_sigma)
        print("Bias Epsilon: \n", self.bias_epsilon)

        # For more detailed output, you could also consider:
        noisy_weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        noisy_bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        print("Noisy Weight: \n", noisy_weight)
        print("Noisy Bias: \n", noisy_bias)

if __name__ == "__main__":
    # Initialize NoisyLinear with example dimensions
    noisy_layer = NoisyLinear(in_features=15, out_features=5)
    # Print the parameters of the NoisyLinear layer
    noisy_layer.print_parameters()
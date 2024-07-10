import math
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class LSTMController(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim, output_activation="tanh"):
        super().__init__()
        self._hidden_size = num_hidden
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self._hidden_size, num_layers=1)
        self.fc = nn.Linear(in_features=self._hidden_size, out_features=output_dim)
        self.activation = nn.Tanh() if output_activation == 'tanh' else nn.Softmax(dim=-1)
        self.reset()
        self.eval()

    def forward(self, x):
        x, self._hidden = self.lstm(x.view(1, 1, -1), self._hidden)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def reset(self):
        self._hidden = (torch.zeros((1, 1, self._hidden_size)), torch.zeros((1, 1, self._hidden_size)))

    def to(self, device):
        super().to(device)
        self._hidden = (self._hidden[0].to(device), self._hidden[1].to(device))

class SelfAttention(nn.Module):
    def __init__(self, data_dim, dim_q):
        super().__init__()
        self.fc_q = nn.Linear(data_dim, dim_q)
        self.fc_k = nn.Linear(data_dim, dim_q)
        self.eval()

    def forward(self, X):
        _, _, K = X.size()
        queries = self.fc_q(X)  # (B, T, Q)
        keys = self.fc_k(X)  # (B, T, Q)
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)
        scaled = torch.div(dot, math.sqrt(K))
        return scaled

class CarRacingAgent(nn.Module):
    def __init__(self, image_size, query_dim, output_dim, output_activation, num_hidden, patch_size, patch_stride, top_k, data_dim, normalize_positions=True):
        super().__init__()
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._top_k = top_k
        self._normalize_positions = normalize_positions

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        n = int((image_size - patch_size) / patch_stride + 1)
        offset = self._patch_size // 2
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * patch_stride
            for j in range(n):
                patch_center_col = offset + j * patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        self._patch_centers = torch.tensor(patch_centers).float()
        self.attention = SelfAttention(data_dim=data_dim * self._patch_size ** 2, dim_q=query_dim)
        self.controller = LSTMController(input_dim=self._top_k * 2, output_dim=output_dim, num_hidden=num_hidden, output_activation=output_activation)
        self.eval()

    def forward(self, x):
        x = x.permute(1, 2, 0)
        _, _, C = x.size()
        patches = x.unfold(0, self._patch_size, self._patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(2, self._patch_size, self._patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self._patch_size, self._patch_size, C))
        flattened_patches = patches.reshape((1, -1, C * self._patch_size ** 2))
        attention_matrix = self.attention(flattened_patches)
        patch_importance_matrix = torch.softmax(attention_matrix.squeeze(), dim=-1)
        patch_importance = patch_importance_matrix.sum(dim=0)
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self._top_k]
        centers = self._patch_centers[top_k_ix].flatten(0, -1).to(self.device)
        if self._normalize_positions:
            centers = centers / self._image_size
        return self.controller(centers).squeeze()

    def step(self, obs):
        with torch.no_grad():
            x = self._transform(obs).to(self.device)
            actions = self.forward(x).cpu().numpy()
        return actions, None

    def reset(self):
        self.controller.reset()
        
    def set_device(self, device):
        self.device = device
        self.to(device)
        self._patch_centers = self._patch_centers.to(device)

def load_pretrained_agent(params_path):
    agent = CarRacingAgent(
        image_size=96,
        query_dim=4,
        output_dim=3,
        output_activation="tanh",
        num_hidden=16,
        patch_size=7,
        patch_stride=4,
        top_k=10,
        data_dim=3,
        normalize_positions=True,
    )
    params = np.load(params_path)['params'].flatten()
    vector_to_parameters(torch.tensor(params), agent.parameters())
    return agent

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

    def create_cross_entropy_loss(self):
        return nn.CrossEntropyLoss()

    def create_sgd_optimizer(self, lr=1e-3):
        return torch.optim.SGD(self.parameters(), lr=lr)

    def create_adam_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)

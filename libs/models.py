import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class BaseModel(nn.Module):
    """Base model for both Actor and Critic"""

    @staticmethod
    def fan_in_initializer(layer):
        """Initializer hidden layer weights as described in DDPG paper"""
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)


class Actor(BaseModel):
    """Actor Model for Policy approoximation."""

    def __init__(self, state_size, action_size, dense_layers=[256, 128], random_state=42):
        """
        Arguments:
            state_size (int) -- Dimension of each state
            action_size (int) -- Dimension of each action
        Keyword Arguments:            
            dense_layers {list} -- Nodes in dense layers (default: {[400, 300]})
            random_state {int} -- seed for torch random number generator (default: {42})
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(random_state)
        self.fc1 = nn.Linear(state_size, dense_layers[0])
        self.fc2 = nn.Linear(dense_layers[0], dense_layers[1])
        self.fc3 = nn.Linear(dense_layers[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.fan_in_initializer(self.fc1))
        self.fc2.weight.data.uniform_(*self.fan_in_initializer(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Mapping of states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(BaseModel):
    """Critic Model for Value approoximation."""

    def __init__(self, state_size, action_size, dense_layers=[256, 128], random_state=42):
        """Arguments:
            state_size (int) -- Dimension of each state
            action_size (int) -- Dimension of each action
        Keyword Arguments:            
            dense_layers {list} -- Nodes in dense layers (default: {[400, 300]})
            random_state {int} -- seed for torch random number generator (default: {42})
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_state)
        self.fc1 = nn.Linear(state_size, dense_layers[0])
        self.fc2 = nn.Linear(dense_layers[0]+action_size, dense_layers[1])
        self.fc3 = nn.Linear(dense_layers[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.fan_in_initializer(self.fc1))
        self.fc2.weight.data.uniform_(*self.fan_in_initializer(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Mapping of (state, action) -> Q-values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

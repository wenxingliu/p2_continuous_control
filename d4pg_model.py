import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import array_to_tensor
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, seed):
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, act_size)
        # self.net = nn.Sequential(
        #     nn.Linear(obs_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, act_size),
        #     nn.Tanh()
        # )
        self.to(device)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def choose_action(self, states):
        actions = self.forward(array_to_tensor(states))
        return actions.cpu().data.numpy()

class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max, seed):
        super(D4PGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.obs_fc = nn.Linear(obs_size, 128)
        self.out_fc1 = nn.Linear(128 + act_size, 64)
        self.out_fc2 = nn.Linear(64, n_atoms)

        # self.obs_net = nn.Sequential(
        #     nn.Linear(obs_size, 128),
        #     nn.ReLU(),
        # )
        #
        # self.out_net = nn.Sequential(
        #     nn.Linear(128 + act_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, n_atoms)
        # )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))
        self.to(device)

    def forward(self, x, a):
        obs = F.leaky_relu(self.obs_fc1(x))
        x = F.leaky_relu(self.out_fc1(torch.cat([obs, a], dim=-1)))
        x = self.out_fc2(x)
        return self.out_net(torch.cat([obs, a], dim=-1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


class D4PGNet(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(D4PGNet, self).__init__()

        self.actor = DDPGActor(state_size, action_size, seed)
        self.critic = D4PGCritic(state_size, action_size, N_ATOMS, Vmin, Vmax, seed)
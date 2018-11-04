from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import torch.optim as optim

from utils import soft_update, ddpg_compute_actor_loss, ddpg_compute_critic_loss, array_to_tensor
from ddpg_model import Actor, Critic, OUNoise
from replay_buffer import AgentMemory
from config import *


class DDPGAgent:

    def __init__(self, num_agents=20, seed=0, train_mode=True):
        self.env = UnityEnvironment(file_name='Reacher_%d.app' % NUM_AGENTS)
        self.brain_name = self.env.brain_names[0]
        self.action_size = 4
        self.state_size = 33
        self.seed = random.seed(seed)
        self.num_agents = num_agents
        self.train_mode = train_mode
        self.max_steps = 1000

        self.step_count = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.scores = np.zeros(self.num_agents)

        self.actor = Actor(self.state_size, self.action_size, seed)
        self.critic = Critic(self.state_size, self.action_size, seed)

        self.noise = OUNoise(self.action_size, seed)
        self.memory = AgentMemory(batch_size=BATCH_SIZE, buffer_size=MEMORY_BUFFER)

        self.target_actor = Actor(self.state_size, self.action_size, seed)
        self.target_critic = Critic(self.state_size, self.action_size, seed)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_C, weight_decay=WEIGHT_DECAY)

    def reset(self):
        self.noise.reset()

        self.scores = np.zeros(self.num_agents)
        self.step_count = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self.states = env_info.vector_observations

    def step(self):
        self.act(add_noise=True)

        env_info = self.env.step(self.actions)[self.brain_name]

        self.next_states = env_info.vector_observations
        self.rewards = env_info.rewards
        self.dones = env_info.local_done
        self.scores += env_info.rewards

        self.step_count += 1
        self.states = self.next_states

        self.memory.add_to_single_memory(self.states, self.actions, self.rewards, self.next_states, self.dones)

        if len(self.memory) >= self.memory.batch_size:
            for _ in range(LEARN_EVERY_20_STEPS):
                states, actions, rewards, next_states, dones = self.memory.sample_single()
                self.learn(states, actions, rewards, next_states, dones)
                self.soft_update()

    def fetch(self, worker_index):
        return (self.states[worker_index],
                self.actions[worker_index],
                self.rewards[worker_index],
                self.next_states[worker_index],
                self.dones[worker_index])

    def act(self, add_noise=True):
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(array_to_tensor(self.states))
            self.actions = actions.cpu().data.numpy()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            self.actions += noise

        self.actions = np.clip(self.actions, -1, 1)

        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        # update critic
        self.critic_opt.zero_grad()
        critic_loss = ddpg_compute_critic_loss(states, actions, rewards, next_states, dones, GAMMA,
                             self.target_actor, self.target_critic, self.critic)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()

        # update actor
        self.actor_opt.zero_grad()
        actor_loss = ddpg_compute_actor_loss(states, self.actor, self.critic)
        actor_loss.backward()
        self.actor_opt.step()

    def soft_update(self):
        soft_update(self.actor, self.target_actor, TAU)
        soft_update(self.critic, self.target_critic, TAU)

    def test_net(self):
        self.train_mode = False
        self.reset()

        while self.step_count < self.max_steps:

            self.act(add_noise=False)
            env_info = self.env.step(self.actions)[self.brain_name]
            self.next_states = env_info.vector_observations
            self.rewards = env_info.rewards
            self.dones = env_info.local_done

            self.scores += env_info.rewards
            self.step_count += 1
            self.states = self.next_states

        self.train_mode = True

    def close(self):
        self.env.close()
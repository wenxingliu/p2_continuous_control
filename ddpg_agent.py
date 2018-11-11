import numpy as np
import torch
import torch.optim as optim

from utils import soft_update, hard_update, ddpg_compute_actor_loss, ddpg_compute_critic_loss, array_to_tensor
from ddpg_model import Actor, Critic, OUNoise
from replay_buffer import AgentMemory
from config import *


class DDPGAgent:

    def __init__(self, action_size=4, state_size=33, num_agents=20, max_steps=1000, seed=0, train_mode=True):
        self.train_mode = train_mode
        self.action_size = action_size
        self.state_size = state_size
        self.num_agents = num_agents
        self.max_steps = max_steps

        self.step_count = 0
        self.scores = np.zeros(self.num_agents)
        self.states, self.actions, self.rewards, self.next_states, self.dones = None, None, None, None, None

        self.noise = OUNoise(self.action_size, seed)
        self.memory = AgentMemory(batch_size=BATCH_SIZE, buffer_size=MEMORY_BUFFER, seed=seed)

        self.actor = Actor(self.state_size, self.action_size, seed)
        self.critic = Critic(self.state_size, self.action_size, seed)

        self.target_actor = Actor(self.state_size, self.action_size, seed)
        self.target_critic = Critic(self.state_size, self.action_size, seed)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        hard_update(self.actor, self.target_actor)
        hard_update(self.critic, self.target_critic)

    def reset(self):
        self.noise.reset()
        self.step_count = 0
        self.scores = np.zeros(self.num_agents)
        self.states, self.actions, self.rewards, self.next_states, self.dones = None, None, None, None, None

    def step(self):
        self.scores += np.array(self.rewards)
        self.step_count += 1
        self.memory.add(self.states, self.actions, self.rewards, self.next_states, self.dones)

        if self.memory.has_enough_memory():
            for i in range(UPDATE_FREQUENCY_PER_STEP):
                states, actions, rewards, next_states, dones = self.memory.sample()
                self.learn(states, actions, rewards, next_states, dones)
                self.soft_update()

    def act(self, add_noise=True):
        states = array_to_tensor(self.states)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
            actions = actions.cpu().data.numpy()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            actions += noise

        actions = np.clip(actions, -1, 1)
        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        # Update critic
        self.critic_opt.zero_grad()
        critic_loss = ddpg_compute_critic_loss(states, actions, rewards, next_states, dones,
                                               self.target_actor, self.target_critic, self.critic)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()


        # Update actor
        self.actor_opt.zero_grad()
        actor_loss = ddpg_compute_actor_loss(states, self.actor, self.critic)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_opt.step()

        # Update target nets
        self.soft_update()

    def soft_update(self):
        soft_update(self.actor, self.target_actor, TAU)
        soft_update(self.critic, self.target_critic, TAU)

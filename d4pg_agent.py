import numpy as np
import random
from utils import array_to_tensor, unpack_trajectories, hard_update, soft_update, d4pg_compute_actor_loss, d4pg_compute_critic_loss
from d4pg_model import D4PGCritic, D4PGActor
from replay_buffer import AgentMemory
import torch
import torch.optim as optim
from config import *


class D4PGAgent:
    
    def __init__(self, seed=0, train_mode=True):
        self.seed = random.seed(seed)
        self.action_size = 4       
        self.state_size = 33
        self.num_agents = 20
        self.train_mode = train_mode
        self.max_steps = 1000

        self.step_count = 0
        self.scores = np.zeros(self.num_agents)
        self.states, self.actions, self.rewards, self.next_states, self.dones = None, None, None, None, None

        self.memory = AgentMemory(batch_size=BATCH_SIZE, buffer_size=MEMORY_BUFFER, seed=seed)

        self.actor = D4PGActor(self.state_size, self.action_size, seed)
        self.critic = D4PGCritic(self.state_size, self.action_size, N_ATOMS, Vmin, Vmax, seed)

        self.target_actor = D4PGActor(self.state_size, self.action_size, seed)
        self.target_critic = D4PGCritic(self.state_size, self.action_size, N_ATOMS, Vmin, Vmax, seed)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_C, weight_decay=WEIGHT_DECAY)

        hard_update(self.actor, self.target_actor)
        hard_update(self.critic, self.target_critic)
    
    def reset(self):
        self.scores = np.zeros(self.num_agents)
        self.step_count = 0
        self.states, self.actions, self.rewards, self.next_states, self.dones = None, None, None, None, None

    def step(self):
        self.scores += np.array(self.rewards)
        self.step_count += 1
        self.memory.add_to_actors(self.states, self.actions, self.rewards, self.next_states, self.dones)

        if self.memory.has_enough_memory():
            for _ in range(UPDATE_FREQUENCY_PER_STEP):
                sampled_trajectories = self.memory.sample_trajectories()
                self.learn(sampled_trajectories)
                self.soft_update()

    def act(self, add_noise=True):
        states = array_to_tensor(self.states)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
            actions = actions.cpu().data.numpy()
        self.actor.train()

        if add_noise:
            actions += np.random.normal(size=actions.shape) * EPSILON
        actions = np.clip(actions, -1, 1)
        return actions

    def fetch(self, worker_index):
        return (self.states[worker_index], 
                self.actions[worker_index], 
                self.rewards[worker_index], 
                self.next_states[worker_index], 
                self.dones[worker_index])

    def learn(self, trajectories):
        states, actions, rewards, next_states, dones = unpack_trajectories(trajectories)

        # update critic
        self.critic_opt.zero_grad()
        critic_loss = d4pg_compute_critic_loss(states, actions, rewards, next_states, dones,
                                               self.target_actor, self.target_critic, self.critic)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()

        # update actor
        self.actor_opt.zero_grad()
        actor_loss = d4pg_compute_actor_loss(states, self.actor, self.critic)
        actor_loss.backward()
        self.actor_opt.step()

    def soft_update(self):
        soft_update(self.actor, self.target_actor, TAU)
        soft_update(self.critic, self.target_critic, TAU)
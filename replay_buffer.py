from collections import deque, namedtuple
import numpy as np
import random
import torch

from utils import array_to_tensor
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class WorkerMemory:
    
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def export(self):
        next_states_buffer, states_buffer, actions_buffer, rewards_buffer = [], [], [], []
        
        for experience in self.memory:
            next_states_buffer.append(experience.next_state)
            states_buffer.append(experience.state)
            actions_buffer.append(experience.action)
            rewards_buffer.append(experience.reward)

        return np.array(next_states_buffer), np.array(states_buffer), np.array(actions_buffer), np.array(rewards_buffer)
    
    def __len__(self):
        return len(self.memory)
    
    def has_enough_memory(self):
        return len(self) >= self.buffer_size
    
    def clear(self):
        self.memory = deque(maxlen=self.buffer_size)


class AgentMemory:

    def __init__(self, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = NUM_AGENTS
        actor_buffer_size = int(buffer_size / self.num_agents)
        self.actor_memories = [WorkerMemory(actor_buffer_size) for i in range(self.num_agents)]
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.trajectory = namedtuple("Trajectory",
                                     field_names=["states", "actions", "rewards", "next_states", "dones"])
        random.seed(seed)

    def add_to_actors(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.actor_memories[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            experience = self.experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.memory.append(experience)

    def sample(self):
        sampled_experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([exp.state for exp in sampled_experiences])
        actions = np.vstack([exp.action for exp in sampled_experiences])
        rewards = np.vstack([exp.reward for exp in sampled_experiences])
        next_states = np.vstack([exp.next_state for exp in sampled_experiences])
        dones = np.vstack([exp.done for exp in sampled_experiences]).astype(np.uint8)

        states = array_to_tensor(states)
        actions = array_to_tensor(actions)
        rewards = array_to_tensor(rewards)
        next_states = array_to_tensor(next_states)
        dones = array_to_tensor(dones)

        return states, actions, rewards, next_states, dones

    def sample_trajectories(self):
        individual_actor_memory_len = len(self.actor_memories[0])

        sampled_indices = random.sample(range(TRAJECTORY_LENGTH * self.num_agents,
                                              individual_actor_memory_len * self.num_agents), k=self.batch_size)
        actor_number = [i % self.num_agents for i in sampled_indices]
        experience_number = [sampled_indices[i] // self.num_agents for i in np.arange(self.batch_size)]

        sampled_trajectories = []

        for i in np.arange(self.batch_size):
            actor_memory = self.actor_memories[actor_number[i]]
            selected_indices = (int(experience_number[i]) - np.arange(TRAJECTORY_LENGTH))[::-1]
            experiences = [actor_memory.memory[e_i] for e_i in selected_indices]
            states = np.vstack([e.state for e in experiences])
            actions = np.vstack([e.action for e in experiences])
            rewards = np.vstack([e.reward for e in experiences]).squeeze(-1)
            next_states = np.vstack([e.next_state for e in experiences])
            dones = np.vstack([e.done for e in experiences]).astype(np.uint8).squeeze(-1)
            trajectory = self.trajectory(states, actions, rewards, next_states, dones)
            sampled_trajectories.append(trajectory)

        return sampled_trajectories

    def has_enough_memory(self):
        if len(self.memory) > 0:
            return len(self.memory) >= self.batch_size
        if len(self.actor_memories[0]) > 0:
            return len(self.actor_memories[0]) * self.num_agents >= self.batch_size + TRAJECTORY_LENGTH * self.num_agents
        return False
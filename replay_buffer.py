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

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = NUM_AGENTS
        # self.actor_memroies = [WorkerMemory(buffer_size) for i in range(self.num_agents)]
        self.actor_memories = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.trajectory = namedtuple("Trajectory",
                                     field_names=["states", "actions", "rewards", "next_states", "dones"])

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.actor_memories[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def add_to_single_memory(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            experience = self.experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.actor_memories.append(experience)

    def sample_single(self):
        sampled_experiences = random.sample(self.actor_memories, k=self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for exp in sampled_experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        states = array_to_tensor(np.array(states))
        actions = array_to_tensor(np.array(actions))
        rewards = array_to_tensor(np.array(rewards))
        next_states = array_to_tensor(np.array(next_states))
        dones = array_to_tensor(np.array(dones).astype(int))

        return states, actions, rewards, next_states, dones

    def sample(self):
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
            states = np.array([e.state for e in experiences])
            actions = np.array([e.action for e in experiences])
            rewards = np.array([e.reward for e in experiences])
            next_states = np.array([e.next_state for e in experiences])
            dones = np.array([e.done for e in experiences])
            trajectory = self.trajectory(states, actions, rewards, next_states, dones)
            sampled_trajectories.append(trajectory)

        return sampled_trajectories

    def has_enough_memory(self):
        return len(self) >= self.batch_size + TRAJECTORY_LENGTH * self.num_agents

    def __len__(self):
        return len(self.actor_memories)
        # return sum([len(m) for m in self.actor_memories])

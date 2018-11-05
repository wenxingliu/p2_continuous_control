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
        # self.actor_memories = [WorkerMemory(buffer_size) for i in range(self.num_agents)]
        self.actor_memories = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.trajectory = namedtuple("Trajectory",
                                     field_names=["states", "actions", "rewards", "next_states", "dones"])
        random.seed(seed)

    def add_to_actors(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.actor_memories[i].add_to_actors(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            experience = self.experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.actor_memories.append(experience)

    def sample(self):
        sampled_experiences = random.sample(self.actor_memories, k=self.batch_size)

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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

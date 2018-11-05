from unityagents import UnityEnvironment
import numpy as np
import random
from utils import array_to_tensor, unpack_trajectories, d4pg_compute_actor_loss, d4pg_compute_critic_loss
import torch
from config import *


class Agent:
    
    def __init__(self, seed=0, num_agents=20, train_mode=True):
        self.seed = random.seed(seed)
        
        self.env = UnityEnvironment(file_name='Reacher_Windows_x86_64\Reacher.exe')
        self.brain_name = self.env.brain_names[0]
        self.action_size = 4       
        self.state_size = 33
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
    
    def reset(self):
        self.scores = np.zeros(self.num_agents)
        self.step_count = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self.states = env_info.vector_observations

    def step(self, local_net, agent_memory):
        local_net.actor.eval()
        with torch.no_grad():
            self.actions = self.collect_actions(local_net)
        local_net.actor.train()

        env_info = self.env.step(self.actions)[self.brain_name]
        self.next_states = env_info.vector_observations
        self.rewards = env_info.rewards
        self.dones = env_info.local_done
        self.scores += env_info.rewards

        self.step_count += 1
        self.states = self.next_states

        agent_memory.add_to_actors(self.states, self.actions, self.rewards, self.next_states, self.dones)
        
    def fetch(self, worker_index):
        return (self.states[worker_index], 
                self.actions[worker_index], 
                self.rewards[worker_index], 
                self.next_states[worker_index], 
                self.dones[worker_index])

    def collect_actions(self, local_net, epsilon=0.3):
        actions = np.array([local_net.actor(array_to_tensor(states)).cpu().data.numpy() for states in self.states])
        noise = np.random.normal(0, 1, size=actions.shape) * epsilon
        actions += noise
        actions = np.clip(actions, -1, 1)
        return actions

    def learn(self, actor_opt, critic_opt, local_net, target_net, trajectories):
        states, actions, rewards, next_states, dones = unpack_trajectories(trajectories)

        # update critic
        critic_opt.zero_grad()
        critic_loss = d4pg_compute_critic_loss(states, actions, rewards, next_states, dones, target_net, local_net)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_net.critic.parameters(), 1)
        critic_opt.step()

        # update actor
        actor_opt.zero_grad()
        actor_loss = d4pg_compute_actor_loss(states, local_net)
        actor_loss.backward()
        actor_opt.step()

    def test_net(self, local_net):
        self.reset()
        while self.step_count < self.max_steps:
            local_net.actor.eval()
            with torch.no_grad():
                self.actions = np.array([local_net.actor(array_to_tensor(states)).cpu().data.numpy()
                                         for states in self.states])
            local_net.actor.train()

            env_info = self.env.step(self.actions)[self.brain_name]
            self.next_states = env_info.vector_observations
            self.rewards = env_info.rewards
            self.dones = env_info.local_done

            self.scores += env_info.rewards
            self.step_count += 1
            self.states = self.next_states
    
    def close(self):
        self.env.close()
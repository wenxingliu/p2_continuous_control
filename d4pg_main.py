from d4pg_agent import Agent
from d4pg_model import D4PGNet as Net
from replay_buffer import AgentMemory
from d4pg_utils import hard_update, soft_update
import torch
import torch.optim as optim

from collections import deque
import numpy as np

from config import *


def train(episodes=100):
    agent = Agent()

    local_net = Net(agent.state_size, agent.action_size)
    target_net = Net(agent.state_size, agent.action_size)

    hard_update(local_net, target_net)

    actor_opt = optim.Adam(local_net.actor.parameters(), lr=LR)
    critic_opt = optim.Adam(local_net.critic.parameters(), lr=LR)

    agent_memory = AgentMemory(batch_size=BATCH_SIZE, buffer_size=MEMORY_BUFFER)

    scores = []
    scores_window = deque(maxlen=100)

    for e in range(1, episodes+1):

        agent.reset()

        for i in range(agent.max_steps):
            agent.step(local_net, agent_memory)

            if agent_memory.has_enough_memory() and (agent.step_count % LEARN_EVERY == 0):
                trajectories = agent_memory.sample()
                agent.learn(actor_opt, critic_opt, local_net, target_net, trajectories)
                soft_update(local_net, target_net, TAU)

        scores.append(agent.scores.mean())
        scores_window.append(agent.scores.mean())

        print('Episode %d, avg score: %.5f' % (e, np.mean(scores_window)))

        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100,
                                                                                         np.mean(scores_window)))
            torch.save(local_net.actor.state_dict(), 'checkpoints/actor_checkpoint_%d.pth' % e)
            torch.save(local_net.critic.state_dict(), 'checkpoints/critic_checkpoint_%d.pth' % e)

    return scores


if __name__ == '__main__':
    scores = train(int(1e4))
    print('breakpoint')
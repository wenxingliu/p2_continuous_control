from collections import deque
import numpy as np
import torch

from ddpg_agent import DDPGAgent
from d4pg_agent import D4PGAgent
from env_wrapper import EnvWrapper

__author__ = 'sliu'

def train_agent(episodes=100, model='DDPG'):

    if model.lower() == 'd4pg':
        agent = D4PGAgent()
        print('Use D4PG agent......\n')
    else:
        agent = DDPGAgent()
        print('Use default DDPG agent......\n')

    env = EnvWrapper(file_name='Reacher_Windows_x86_64\Reacher.exe', train_mode=True)

    scores = []
    scores_window = deque(maxlen=100)

    for ep in range(1, episodes + 1):
        agent.reset()
        agent.states = env.reset()

        for s in range(agent.max_steps):
            agent.actions = agent.act(add_noise=True)
            agent.rewards, agent.next_states, agent.dones = env.step(agent.actions)
            agent.step()
            agent.states = agent.next_states

        scores.append(agent.scores.mean())
        scores_window.append(agent.scores.mean())

        print('Episode %d, avg score: %.2f' % (ep, agent.scores.mean()))

        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor.state_dict(), 'checkpoints/actor_checkpoint_%d.pth' % ep)
            torch.save(agent.critic.state_dict(), 'checkpoints/critic_checkpoint_%d.pth' % ep)

    env.close()

    return scores


if __name__ == '__main__':
    train_agent(200, 'D4PG')
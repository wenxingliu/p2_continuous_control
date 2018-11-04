from collections import deque
import numpy as np
import torch

from ddpg_agent import DDPGAgent

__author__ = 'sliu'

def train(episodes=100):

    agent = DDPGAgent()

    scores = []
    scores_window = deque(maxlen=100)

    for ep in range(1, episodes + 1):
        agent.reset()

        for s in range(agent.max_steps):
            agent.step()

        scores.append(agent.scores.mean())
        scores_window.append(agent.scores.mean())

        print('Episode %d, avg score: %.2f' % (ep, agent.scores.mean()))

        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor.state_dict(), 'checkpoints/actor_checkpoint_%d.pth' % ep)
            torch.save(agent.critic.state_dict(), 'checkpoints/critic_checkpoint_%d.pth' % ep)



if __name__ == '__main__':
    train(300)
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def array_to_tensor(arr):
    return torch.from_numpy(arr).float().to(device)


def hard_update(local_netowrks, target_networks):
    for target_param, local_param in zip(target_networks.parameters(), local_netowrks.parameters()):
        target_param.data.copy_(local_param.data)


def soft_update(local_netowrks, target_networks, tau):
    for target_param, local_param in zip(target_networks.parameters(), local_netowrks.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def unpack_trajectories(trajectories):
    states = array_to_tensor(np.array([trajectory.states for trajectory in trajectories]))
    actions = array_to_tensor(np.array([trajectory.actions for trajectory in trajectories]))
    rewards = array_to_tensor(np.array([trajectory.rewards for trajectory in trajectories]))
    next_states = array_to_tensor(np.array([trajectory.next_states for trajectory in trajectories]))
    dones = np.array([trajectory.dones for trajectory in trajectories])
    return states, actions, rewards, next_states, dones


def distr_projection(next_distr, rewards, dones_mask):
    gamma = GAMMA ** TRAJECTORY_LENGTH
    proj_distr = np.zeros((BATCH_SIZE, TRAJECTORY_LENGTH, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return array_to_tensor(proj_distr)


def d4pg_compute_critic_loss(states, actions, rewards, next_states, dones, target_actor, target_critic, local_critic):
    crt_distr_v = local_critic(states, actions)
    last_act_v = target_actor(next_states)
    last_distr_v = F.softmax(target_critic(next_states, last_act_v), dim=1)

    last_distr = last_distr_v.cpu().data.numpy()
    rewards = rewards.cpu().data.numpy()

    proj_dist = distr_projection(last_distr, rewards, dones)
    prob_dist = - F.log_softmax(crt_distr_v, dim=1) * proj_dist

    critic_loss = prob_dist.sum(dim=1).mean()
    return critic_loss


def d4pg_compute_actor_loss(states, local_actor, local_critic):
    predict_actions = local_actor(states)
    crt_distr_v = local_critic(states, predict_actions)
    actor_loss_v = - local_critic.distr_to_q(crt_distr_v)
    actor_loss = actor_loss_v.mean()
    return actor_loss


def ddpg_compute_critic_loss(states, actions, rewards, next_states, dones,
                             target_actor, target_critic, local_critic):
    actions_next = target_actor(next_states)
    target_q_next = target_critic(next_states, actions_next)
    # Compute Q targets for current states (y_i)
    target_q = rewards + (GAMMA * target_q_next * (1 - dones))
    # Compute critic loss
    local_q_pred = local_critic(states, actions)
    critic_loss = F.mse_loss(local_q_pred, target_q)
    return critic_loss


def ddpg_compute_actor_loss(states, local_actor, local_critic):
    actions_pred = local_actor(states)
    actor_loss = - local_critic(states, actions_pred).mean()
    return actor_loss


def test_agent(env, agent, test_episodes=10):
    env.train_mode = False
    scores = []
    for i in range(1, test_episodes+1):

        agent.reset()
        env.reset()
        agent.states = env.reset()
        done = False
        while not done:
            agent.act(add_noise=False)
            agent.rewards, agent.next_states, agent.dones = env.step(agent.actions)
            agent.scores += agent.rewards
            agent.step_count += 1
            agent.states = agent.next_states
            done = any(agent.dones)

        print('Episode %d, avg score %.2f' % (i, agent.scores.mean()))

        scores.append(agent.scores.mean())

    return scores


def plot_scores(scores, model_name='DDPG'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()

    fig.savefig('%s_scores.png' % model_name)
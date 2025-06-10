import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from grag_turn_env import GRAGTurnEnv
from models import DQN
from replay_buffer import ReplayBuffer
from logger import EpisodeLogger
import os

EPISODES = 3000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_SIZE = 10000
TARGET_UPDATE = 10

epsilon = EPSILON_START

graph = nx.erdos_renyi_graph(5, 0.6)
env = GRAGTurnEnv(graph)
obs_shape = env.n_nodes * 2
action_space = env.n_nodes

def save_model(model, episode, path="models"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "episode": episode,
        "epsilon": epsilon
    }, f"{path}/model_ep{episode}.pt")

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(obs_shape, action_space, env.max_resources).to(device)
target_net = DQN(obs_shape, action_space, env.max_resources).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_SIZE)

from simulation import GRAGVisualizer 

vis = GRAGVisualizer(graph)


logger = EpisodeLogger("results/turn_episode_log.csv")

def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(0, action_space, size=env.max_resources)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state).view(env.max_resources, action_space)
            return q_values.argmax(dim=1).cpu().numpy()

def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).view(BATCH_SIZE, env.max_resources, action_space)
    chosen_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    next_q_values = target_net(next_states).view(BATCH_SIZE, env.max_resources, action_space)
    next_max_q = next_q_values.max(dim=2)[0]

    target = rewards + GAMMA * next_max_q * (1 - dones)
    loss = F.mse_loss(chosen_q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

SAVE_INTERVAL = 50
episode_rewards = []
avg_rewards = []

for ep in range(EPISODES):
    env.reset()
    total_reward = 0
    step = 0
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    last_obs = None
    last_action = None

    for agent in env.agent_iter():
        obs = env.observe(agent)

        if env.terminations.get(agent, True):
            action = None
        else:
            action = select_action(obs)

        if agent == "player_1":
            reward = env.rewards.get(agent, 0)
            done = env.terminations.get(agent, False)
            next_obs = obs 

            if last_obs is not None and last_action is not None:
                replay_buffer.push(last_obs, last_action, reward, next_obs, done)
                logger.log_step(ep, step, last_obs, last_action, reward, next_obs, done)
                train_step()
                total_reward += reward
                step += 1

            last_obs = obs
            last_action = action

        env.step(action)

        rl_info = {
            'episode': ep,
            'step': step,
            'reward': total_reward,
            'epsilon': epsilon
        }
        vis.draw(env.resources, active_agent=agent, rl_info=rl_info)

        
        if agent == "player_1" and not env.terminations.get(agent, True):
            reward = env.rewards.get(agent, 0)
            done = env.terminations.get(agent, True)
            next_obs = env.observe(agent)
            replay_buffer.push(obs, action, reward, next_obs, done)
            logger.log_step(ep, step, obs, action, reward, next_obs, done)
            train_step()
            total_reward += reward
            step += 1


    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    logger.log_episode(total_reward)
    episode_rewards.append(total_reward)
    if len(episode_rewards) >= 10:
        avg_rewards.append(np.mean(episode_rewards[-10:]))

    if ep % SAVE_INTERVAL == 0:
        save_model(policy_net, ep)
        EpisodeLogger.save_rewards(episode_rewards, avg_rewards, logger.cumulative_rewards)

    print(f"[Ep {ep}] Reward: {total_reward:.2f}, ε={epsilon:.3f}")

vis.show_final()

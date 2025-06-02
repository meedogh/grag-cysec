import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from grag_env import GRAGEnv
from models import DQN
from replay_buffer import ReplayBuffer
from logger import EpisodeLogger
import os

EPISODES = 2000
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
env = GRAGEnv(graph)
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
target_net = load_model(policy_net, "models/model_ep1950.pt")
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_SIZE)

logger = EpisodeLogger("results/episode_log.csv")

def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(0, action_space, size=env.max_resources)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)  # shape: [1, max_resources * n_nodes]
            q_values = q_values.view(env.max_resources, action_space)  # shape: [R, A]
            return q_values.argmax(dim=1).cpu().numpy()




def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.FloatTensor(states).to(device) 
    actions = torch.LongTensor(actions).to(device)     # [B, max_resources]
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # [B, 1]
    next_states = torch.FloatTensor(next_states).to(device)  
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  

    assert torch.max(actions) < action_space, f"Invalid action index {torch.max(actions)} >= {action_space}"

    rewards = torch.clamp(rewards, -1.0, 1.0)

    q_values = policy_net(states).view(BATCH_SIZE, env.max_resources, action_space)  # [B, R, A]
    # print("q_values.shape:", q_values.shape)
    # print("actions.shape:", actions.shape)
    # print("actions.unsqueeze(-1).shape:", actions.unsqueeze(-1).shape)

    chosen_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, R]

    next_q_values = target_net(next_states).view(BATCH_SIZE, env.max_resources, action_space)
    next_max_q = next_q_values.max(dim=2)[0]  # [B, R]

    target = rewards + GAMMA * next_max_q * (1 - dones)  # [B, R]

    loss = F.mse_loss(chosen_q_values, target)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

    optimizer.step()



SAVE_INTERVAL = 50
episode_rewards = []
avg_rewards = []


for ep in range(EPISODES):
    obs = env.reset()["player_1"]
    done = False
    step = 0

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    total_reward = 0
    while not done:

        if ep % 10 == 0 and step == 0:
            avg_reward = np.mean(logger.rewards[-10:]) if len(logger.rewards) >= 10 else 0
            print(f"[Ep {ep}] Avg Reward: {avg_reward:.3f}, ε={epsilon:.3f}")

        if ep % 50 == 0 and step == 0:
            save_model(policy_net, ep)

        action = select_action(obs)
        action_vector = action  # already shape (max_resources,)

        if np.any(action_vector >= action_space):
            print("Invalid action:", action_vector)
            continue


        random_action = np.random.randint(0, env.n_nodes, size=env.max_resources)
        actions = {"player_1": action, "player_2": random_action}

        next_obs, rewards, dones, _ = env.step(actions)

        reward = rewards["player_1"]
        done = dones["__all__"]
        next_obs = next_obs["player_1"]

        replay_buffer.push(obs, action_vector, reward, next_obs, done)
        logger.log_step(ep, step, obs, action_vector, reward, next_obs, done)


        obs = next_obs
        total_reward += reward
        step += 1

        train_step()

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


    logger.log_episode(total_reward)
    episode_rewards.append(total_reward)

    if len(episode_rewards) >= 10:
        avg_10 = np.mean(episode_rewards[-10:])
        avg_rewards.append(avg_10)
        
    if ep % SAVE_INTERVAL == 0 and ep > 0:
        EpisodeLogger.save_rewards(episode_rewards, avg_rewards, logger.cumulative_rewards)

        
    logger.log_episode(total_reward)
    if len(logger.rewards) >= 10:
        avg_reward = np.mean(logger.rewards[-10:])
        # print(f"Avg reward over last 10 episodes: {avg_reward}")
    print(f"Step {step}, Action: {action_vector}, Reward: {reward}")
    print(f"Episode {ep}, ε={epsilon:.3f}, Total Reward: {total_reward:.2f}")


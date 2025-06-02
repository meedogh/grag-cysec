import pickle
import matplotlib.pyplot as plt

def load_rewards(filename_prefix="results/rewards"):
    with open(f"{filename_prefix}_episode.pkl", "rb") as f:
        episode_rewards = pickle.load(f)
    with open(f"{filename_prefix}_avg.pkl", "rb") as f:
        avg_rewards = pickle.load(f)
    with open(f"{filename_prefix}_cumulative.pkl", "rb") as f:
        cumulative_rewards = pickle.load(f)
    return episode_rewards, avg_rewards, cumulative_rewards

episode_rewards, avg_rewards, cumulative_rewards = load_rewards()

fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

axs[0].plot(episode_rewards, label="Episode Reward", color='blue')
axs[0].set_ylabel("Reward")
axs[0].set_title("Episode Reward Over Episodes")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(9, 9 + len(avg_rewards)), avg_rewards, label="Average Reward (per 10 episodes)", color='green', linewidth=3)
axs[1].set_ylabel("Average Reward")
axs[1].set_title("Average Reward Over Episodes")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(cumulative_rewards, label="Cumulative Reward", color='red')
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Cumulative Reward")
axs[2].set_title("Cumulative Reward Over Episodes")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

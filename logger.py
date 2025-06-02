import csv
import pickle
import os
import numpy as np

class EpisodeLogger:
    def __init__(self, filename="episode_log.csv"):
        self.filename = filename
        self.fields = ["episode", "step", "state", "action", "reward", "next_state", "done"]
        self.rewards = [] 
        self.cumulative_rewards = []  

        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log_step(self, episode, step, state, action, reward, next_state, done):
        with open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                "episode": episode,
                "step": step,
                "state": state.tolist(),
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
                "reward": reward,
                "next_state": next_state.tolist(),
                "done": done
            })

    def log_episode(self, total_reward):
        self.rewards.append(total_reward)
        cumulative = self.cumulative_rewards[-1] + total_reward if self.cumulative_rewards else total_reward
        self.cumulative_rewards.append(cumulative)

    @staticmethod
    def save_rewards(rewards, avg_rewards, cumulative_rewards=None, filename_prefix="results/rewards"):
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
        with open(f"{filename_prefix}_episode.pkl", "wb") as f:
            pickle.dump(rewards, f)
        with open(f"{filename_prefix}_avg.pkl", "wb") as f:
            pickle.dump(avg_rewards, f)
        if cumulative_rewards is not None:
            with open(f"{filename_prefix}_cumulative.pkl", "wb") as f:
                pickle.dump(cumulative_rewards, f)



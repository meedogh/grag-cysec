import numpy as np
import networkx as nx
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces

class GRAGEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "grag_v0"}

    def __init__(self, graph: nx.Graph, total_resources: int = 8):
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.total_resources = total_resources

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.max_resources = total_resources
        self.action_spaces = {
            agent: spaces.MultiDiscrete([self.n_nodes] * self.max_resources)
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=-total_resources, high=total_resources, shape=(self.n_nodes,), dtype=np.int32)
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.steps = 0
        self.done = False

        self.resources = {
            "player_1": self._init_distribution(),
            "player_2": self._init_distribution()
        }

        return self.observe()

    def _init_distribution(self):
        res = np.zeros(self.n_nodes, dtype=np.int32)
        indices = np.random.choice(self.n_nodes, self.total_resources)
        for i in indices:
            res[i] += 1
        return res

    def observe(self):
        return {
            "player_1": np.concatenate([self.resources["player_1"], self.resources["player_2"]]),
            "player_2": np.concatenate([self.resources["player_2"], self.resources["player_1"]])
        }


    def step(self, actions):
        self.steps += 1
        rewards = {"player_1": 0, "player_2": 0}
        terminations = {"__all__": False}

        new_resources = {agent: np.zeros_like(self.resources[agent]) for agent in self.agents}

        for agent in self.agents:
            res = self.resources[agent]
            action = actions[agent]
            resource_idx = 0

            for node in range(self.n_nodes):
                for _ in range(res[node]):
                    if resource_idx >= self.max_resources:
                        break
                    
                    target = int(action[resource_idx].item())

                    if target == node or self.graph.has_edge(node, target):
                        new_resources[agent][target] += 1
                    else:
                        new_resources[agent][node] += 1 
                    resource_idx += 1

        self.resources = new_resources

        control_diff = self.resources["player_1"] - self.resources["player_2"]
        p1_win = np.sum(control_diff > 0)
        p2_win = np.sum(control_diff < 0)

        territory_advantage = np.sum(control_diff > 0) - np.sum(control_diff < 0)
        rewards["player_1"] = territory_advantage / self.n_nodes  # Normalize [-1,1]
        rewards["player_2"] = -rewards["player_1"]
        
        reward_p1 = 0
        if p1_win + p2_win >= self.n_nodes:
            if p1_win > p2_win:
                reward_p1 = 1
            elif p2_win > p1_win:
                reward_p1 = -1
            terminations["__all__"] = True

        rewards["player_1"] = reward_p1
        rewards["player_2"] = -reward_p1


        return self.observe(), rewards, terminations, {}

    def render(self):
        print("Current Resource State:")
        for agent in self.agents:
            print(f"{agent}: {self.resources[agent]}")

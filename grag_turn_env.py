import numpy as np
import networkx as nx
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector as AgentSelector
from gymnasium import spaces


class GRAGTurnEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "grag_turn_v0"}

    def __init__(self, graph: nx.Graph, total_resources: int = 8):
        super().__init__()
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.total_resources = total_resources
        self.max_resources = total_resources

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.agent_selector = AgentSelector(self.agents)
        self._agent_iterator = None

        self.action_spaces = {
            agent: spaces.MultiDiscrete([self.n_nodes] * self.max_resources)
            for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Box(low=-total_resources, high=total_resources, shape=(self.n_nodes * 2,), dtype=np.int32)
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agent_selector.reset()  # returns first agent string
        self.steps = 0
        self.done = False
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.resources = {
            "player_1": self._init_distribution(),
            "player_2": self._init_distribution()
        }

        self.new_resources = {
            "player_1": np.zeros(self.n_nodes, dtype=np.int32),
            "player_2": np.zeros(self.n_nodes, dtype=np.int32)
        }

        self._actions = {agent: None for agent in self.agents}


    def _init_distribution(self):
        res = np.zeros(self.n_nodes, dtype=np.int32)
        indices = np.random.choice(self.n_nodes, self.total_resources)
        for i in indices:
            res[i] += 1
        return res

    def observe(self, agent):
        self._check_agent(agent)
        return np.concatenate([self.resources[agent], self.resources[self._opponent(agent)]])

    def _opponent(self, agent):
        return "player_2" if agent == "player_1" else "player_1"

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent]:
            self._was_dead_step(action)
            return

        self._actions[agent] = action

        if all(a is not None for a in self._actions.values()):
            self._resolve_step()

        self.agent_selection = self.agent_selector.next()  # get next agent string


    def _resolve_step(self):
        for agent in self.agents:
            res = self.resources[agent]
            action = self._actions[agent]
            resource_idx = 0
            for node in range(self.n_nodes):
                for _ in range(res[node]):
                    if resource_idx >= self.max_resources:
                        break
                    target = int(action[resource_idx])
                    if target == node or self.graph.has_edge(node, target):
                        self.new_resources[agent][target] += 1
                    else:
                        self.new_resources[agent][node] += 1
                    resource_idx += 1

        self.resources = {
            "player_1": self.new_resources["player_1"].copy(),
            "player_2": self.new_resources["player_2"].copy()
        }

        control_diff = self.resources["player_1"] - self.resources["player_2"]
        p1_win = np.sum(control_diff > 0)
        p2_win = np.sum(control_diff < 0)

        if p1_win + p2_win >= self.n_nodes:
            reward_p1 = 1 if p1_win > p2_win else -1 if p2_win > p1_win else 0
            self.rewards["player_1"] = reward_p1
            self.rewards["player_2"] = -reward_p1
            self.terminations = {agent: True for agent in self.agents}

        self._actions = {agent: None for agent in self.agents}
        self.new_resources = {
            "player_1": np.zeros(self.n_nodes, dtype=np.int32),
            "player_2": np.zeros(self.n_nodes, dtype=np.int32)
        }

    
    def _check_agent(self, agent):
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not in the environment.")


    def render(self):
        print("=== Turn-Based GRAG State ===")
        for agent in self.agents:
            print(f"{agent}: {self.resources[agent]}")


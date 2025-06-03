import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import time
import numpy as np

class GRAGVisualizer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.pos = nx.spring_layout(graph, seed=42)

    def draw(self, resources: np.ndarray, active_agent=None, pause=0.5, rl_info=None):
        plt.clf()
        ax = plt.gca()

        nx.draw_networkx_edges(self.graph, pos=self.pos, edge_color='#37474F')
        if rl_info:
            info_text = "\n".join([
                f"Episode: {rl_info.get('episode', '?')}",
                f"Step: {rl_info.get('step', '?')}",
                f"Reward: {rl_info.get('reward', 0):.2f}",
                f"Epsilon: {rl_info.get('epsilon', 0):.3f}"
            ])
            plt.gcf().text(0.01, 0.95, info_text, fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

        for node in self.graph.nodes:
            p1_res = resources['player_1'][node]
            p2_res = resources['player_2'][node]

            total = p1_res + p2_res
            x, y = self.pos[node]

            if total > 0:
                sizes = [p1_res / total, p2_res / total]
                colors = ['#4CAF50', '#E57373']  # green for P1, red for P2
                wedges, _ = ax.pie(sizes, colors=colors, radius=0.15, center=(x, y))
                for wedge in wedges:
                    wedge.set_edgecolor('black')
                    wedge.set_linewidth(0.7)
            else:
                circle = plt.Circle((x, y), 0.15, color='#B0BEC5', ec='black', lw=1.2)
                ax.add_patch(circle)

            label = f"{node}\nP1: {p1_res}\nP2: {p2_res}"
            plt.text(x, y - 0.25, label, fontsize=10, ha='center', va='top', fontweight='bold')

        if active_agent == 'player_1':
            plt.title("GRAG Environment - Player 1's Turn", fontsize=16, color='#4CAF50')
        elif active_agent == 'player_2':
            plt.title("GRAG Environment - Player 2's Turn", fontsize=16, color='#E57373')
        else:
            plt.title("GRAG Environment Resource Control", fontsize=16)

        # Legend for control colors
        green_patch = mpatches.Patch(color="#4CAF50", label='Player 1 Control')
        red_patch = mpatches.Patch(color="#E57373", label='Player 2 Control')
        gray_patch = mpatches.Patch(color="#B0BEC5", label='Tie / Neutral')
        plt.legend(handles=[green_patch, red_patch, gray_patch], loc='upper right')

        plt.axis('equal')  
        plt.axis('off')   
        plt.pause(pause)

    def show_final(self):
        plt.show()



if __name__ == "__main__":
    from grag_turn_env import GRAGTurnEnv
    graph = nx.erdos_renyi_graph(5, 0.6)
    env = GRAGTurnEnv(graph)
    vis = GRAGVisualizer(graph)

    env.reset()
    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            action = None
        else:
            action = np.random.randint(0, env.n_nodes, size=env.max_resources)
        env.step(action)
        vis.draw(env.resources, active_agent=agent)

    vis.show_final()

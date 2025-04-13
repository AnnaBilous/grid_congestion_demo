import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_lv_grid(n_houses=30, seed=42):
    """
    Generate a tree-like low-voltage grid with 1 transformer and n_houses.
    Returns a NetworkX graph.
    """
    random.seed(seed)
    G = nx.random_tree(n_houses + 1, seed=seed)  # +1 for the transformer
    mapping = {0: 'transformer'}
    mapping.update({i: f'house_{i}' for i in range(1, n_houses + 1)})
    G = nx.relabel_nodes(G, mapping)
    return G

def plot_lv_grid(G):
    """
    Plot the low-voltage network.
    """
    pos = nx.spring_layout(G, seed=1)
    node_colors = ['red' if n == 'transformer' else 'skyblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8)
    plt.title("Synthetic Low-Voltage Grid")
    plt.show()

# Example usage
if __name__ == "__main__":
    grid = generate_lv_grid(n_houses=30)
    plot_lv_grid(grid)

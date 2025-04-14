import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_lv_grid(n_houses=30, seed=None):
    """
    Generate a tree-like low-voltage grid with 1 transformer and n houses.
    
    The grid is represented as a tree where the transformer is the root node
    and houses are connected through a realistic distribution network topology.
    
    Parameters
    ----------
    n_houses : int, default=30
        Number of houses to connect to the grid
    seed : int, default=None
        Random seed for reproducibility
        
    Returns
    -------
    networkx.Graph
        A tree graph with n_houses + 1 nodes (including transformer)
    """
    if not seed:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    G = nx.random_tree(n_houses + 1, seed=seed)  # +1 for the transformer
    mapping = {0: 'transformer'}
    mapping.update({i: f'house_{i}' for i in range(1, n_houses + 1)})
    G = nx.relabel_nodes(G, mapping)
    return G

def plot_lv_grid(G, plot_name=None):
    """
    Plot the low-voltage network with the transformer highlighted.
    
    Parameters
    ----------
    G : networkx.Graph
        The grid graph to visualize
    plot_name : str, optional
        If provided, save the plot to this path instead of displaying
        
    Returns
    -------
    None
        Either displays the plot or saves it to file
    """
    pos = nx.spring_layout(G, seed=1)
    node_colors = ['red' if n == 'transformer' else 'skyblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8)
    plt.title("Synthetic Low-Voltage Grid")
    if plot_name:
        plt.savefig(plot_name)
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    grid = generate_lv_grid(n_houses=30)
    plot_lv_grid(grid)

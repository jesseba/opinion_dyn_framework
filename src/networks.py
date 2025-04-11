import networkx as nx

def create_scale_free_network(n=1000, m=3):
    """Create a scale-free network using Barabási-Albert model."""
    return nx.barabasi_albert_graph(n, m)

def create_small_world_network(n=1000, k=6, p=0.1):
    """Create a small-world network using Watts-Strogatz model."""
    return nx.watts_strogatz_graph(n, k, p)

def create_random_network(n=1000, p=None, k=6):
    """Create a random Erdős-Rényi network with similar average degree."""
    if p is None:
        # Calculate p to achieve target average degree k
        p = k / (n - 1)
    return nx.erdos_renyi_graph(n, p) 
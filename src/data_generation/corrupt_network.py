import numpy as np
import networkx as nx
import random


def make_corrupt_network(
    adj_matrix: np.array,
    edge_removal_prob: float = 0.15,
    edge_addition_prob: float = 0.03,
    edge_swap_frac: float = 0.1,
    node_dropout_prob: float = 0.0,
    ensure_connected: bool = True,
    seed: int = 12345
):
    """Make partial observability of 

    Args:
        adj_matrix (np.array): _description_
        edge_removal_prob (float, optional): _description_. Defaults to 0.15.
        edge_addition_prob (float, optional): _description_. Defaults to 0.03.
        edge_swap_frac (float, optional): _description_. Defaults to 0.1.
        node_dropout_prob (float, optional): _description_. Defaults to 0.0.
        ensure_connected (bool, optional): _description_. Defaults to True.
        seed (int, optional): _description_. Defaults to 12345.

    Returns:
        _type_: _description_
    """    
    G = nx.from_numpy_array(adj_matrix)
    nodes = list(G.nodes())

    # 1. Remove edges
    edges = list(G.edges())
    num_remove = int(len(edges) * edge_removal_prob)
    if num_remove > 0:
        remove_edges = np.random.choice(len(edges), num_remove, replace=False)
        for idx in remove_edges:
            u, v = edges[idx]
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    # 2. Add random edges
    edges = list(G.edges())
    num_add = int(len(edges) * edge_addition_prob)
    potential_edges = [
        (u, v) for u in nodes for v in nodes if u < v and not G.has_edge(u, v)
    ]
    if num_add > 0 and potential_edges:
        new_edges = np.random.choice(len(potential_edges), min(num_add, len(potential_edges)), replace=False)
        for idx in new_edges:
            u, v = potential_edges[idx]
            G.add_edge(u, v)

    # 3. Swap edges
    num_swap = int(G.number_of_edges() * edge_swap_frac)
    swap_count = 0
    attempts = 0
    while swap_count < num_swap and attempts < num_swap * 10:
        edges = list(G.edges())
        if len(edges) < 2:
            break
        e1, e2 = random.sample(edges, 2)
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) == 4:
            if not G.has_edge(a, d) and not G.has_edge(c, b):
                if G.has_edge(a, b) and G.has_edge(c, d):
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)
                    G.add_edge(a, d)
                    G.add_edge(c, b)
                    swap_count += 1
        attempts += 1

    # 4. Drop nodes
    num_nodes_to_drop = int(len(nodes) * node_dropout_prob)
    if num_nodes_to_drop > 0:
        nodes_to_remove = np.random.choice(nodes, num_nodes_to_drop, replace=False)
        G.remove_nodes_from(nodes_to_remove)

    # 5. Ensure that no nodes are isolated
    if ensure_connected:
        for node in G.nodes():
            if G.degree(node) == 0:
                candidates = [n for n in G.nodes() if n != node]
                if candidates:
                    partner = np.random.choice(candidates)
                    G.add_edge(node, partner)

    A_corrupt = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    return A_corrupt

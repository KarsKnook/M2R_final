import networkx as nx
import numpy as np
import random


def generate_graph(n=10, a=0, b=1):
    # Generates complete weighted graph with n nodes and edge weights uniformly distributed between a and b.
    # Triangle equality does not hold!
    G = nx.Graph()
    edge_list = [(i, j, random.uniform(a, b)) for i in range(n-1) for j in range(i, n)] 
    G.add_weighted_edges_from(edge_list)
    return G


def generate_graph_triangle(n=10, a=1, b=1):
    # Generates complete weighted graph with n nodes with position in [0, a]x[0, b]
    # Outputs complete weighted graph and a dictionary of positions which is helpful for drawing the graph
    # Triangle equality does hold!
    G = nx.Graph()

    node_labels = range(n)
    coordinates = [np.array([random.uniform(0, a), random.uniform(0, b)]) for i in node_labels]
    pos = dict(zip(node_labels, coordinates))
    
    edge_list = [(i, j, np.linalg.norm(coordinates[i] - coordinates[j])) for i in range(n-1) for j in range(i, n)] 
    G.add_weighted_edges_from(edge_list)

    return G, pos


def generate_graph_triangle_3D(n=10, a=1, b=1, c=1):
    # Generates complete weighted graph with n nodes with position in [0, a]x[0, b]x[0, c]
    # Outputs complete weighted graph and a dictionary of positions which is helpful for drawing the graph
    # Triangle equality does hold!
    G = nx.Graph()

    node_labels = range(n)
    coordinates = [np.array([random.uniform(0, a), random.uniform(0, b), random.uniform(0, c)]) for i in node_labels]
    pos = dict(zip(node_labels, coordinates))
    
    edge_list = [(i, j, np.linalg.norm(coordinates[i] - coordinates[j])) for i in range(n-1) for j in range(i, n)] 
    G.add_weighted_edges_from(edge_list)

    return G, pos

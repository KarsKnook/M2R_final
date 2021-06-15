import networkx as nx
import numpy as np
import random


def generate_graph(n=10, a=0, b=1):
    """
    Input: n is the number of nodes we want
    Edge weights are sampled from the uniform distribution U([a,b])
    Output: G is a complete weighted graph
    """
    G = nx.Graph() # create a graph
    edge_list = [(i, j, random.uniform(a, b)) for i in range(n-1) for j in range(i, n)] 
    G.add_weighted_edges_from(edge_list)
    return G


def generate_graph_triangle(n=10, a=1, b=1):
    """
    Input: n is the number of nodes we want
    [0,a] x [0,b] is the set where the nodes lie within
    Output: G is a complete weighted graph with n nodes with positions 
    randomly generated from the uniform distribution U([0, a]x[0, b])
    with weight equal to the distance between the two nodes
    the weight is a metric!
    """
    G = nx.Graph()  # create a graph

    node_labels = range(n)  # label the n nodes
    # generate the position coordinates of the nodes
    coordinates = [np.array([random.uniform(0, a), random.uniform(0, b)]) for i in node_labels]
    pos = dict(zip(node_labels, coordinates))
    
    # define the weight of an edge by the Euclidean distance between the two nodes
    edge_list = [(i, j, np.linalg.norm(coordinates[i] - coordinates[j])) for i in range(n-1) for j in range(i, n)] 
    G.add_weighted_edges_from(edge_list)

    return G, pos


def generate_graph_triangle_3D(n=10, a=1, b=1, c=1):
    """
    Input: n is the number of nodes we want
    [0,a] x [0,b] x [0,c] is the set where the nodes lie within
    Output: G is a complete weighted graph with n nodes with positions 
    randomly generated from the uniform distribution U([0, a]x[0, b]x[0,c])
    with weight equal to the distance between the two nodes
    the weight is a metric!
    """
    G = nx.Graph()  # create a graph

    node_labels = range(n)  # label the n nodes
    # generate the position coordinates of the nodes
    coordinates = [np.array([random.uniform(0, a), random.uniform(0, b), random.uniform(0, c)]) for i in node_labels]
    pos = dict(zip(node_labels, coordinates))
    
    # define the weight of an edge by the Euclidean distance between the two nodes
    edge_list = [(i, j, np.linalg.norm(coordinates[i] - coordinates[j])) for i in range(n-1) for j in range(i, n)] 
    G.add_weighted_edges_from(edge_list)

    return G, pos

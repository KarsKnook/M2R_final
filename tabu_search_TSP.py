import networkx as nx
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import itertools
import math
from geopy import Nominatim


def get_coordinates(cities):
    geolocator = Nominatim(user_agent="tabu_search_application")
    cities_coordinates = []
    
    for city in cities:
        try:
            location = geolocator.geocode(city)
            cities_coordinates.append((location.longitude, location.latitude))
        except:
            raise Exception(f"There was a problem with the coordinates of {city}")
    return cities_coordinates


def roulette(initial_list):
    """
        Input: list of positive numbers.
        Output: A 2-tuple containing a random element from the Python list other than the maximum and 
        its index.
    """

    # remove the minimum of initial_list
    second_list = initial_list.copy()
    minimum = min(second_list)
    second_list.remove(minimum)

    # assign a probability distribution to all non-minimum elements in initial_list
    # the probability of getting x should be inversely proportional to x
    reciprocals = np.array([1/x for x in second_list])
    probabilities = reciprocals/np.sum(reciprocals)

    # draw x from the roulette distribution and find the index of x
    random_number = np.random.choice(second_list, p=probabilities) 
    number_index = initial_list.index(random_number)

    return random_number, number_index


def length(G, path):
    """
     G is a weighted connected graph, preferably with a small (<50) number of nodes.
     path is a list of n-1 city indexes (excluding fixed point 0)
     output: the total length of the path
    """
    n = G.number_of_nodes()
    distance = sum([G[path[i]][path[i+1]]['weight'] for i in range(n - 2)]) + G[path[-1]][0]['weight'] + G[path[0]][0]['weight']
    return distance

 
def greedy_path(G):
    """
    Input: G is a complete weighted undirected graph
    Output: The greedy circular path starting in 0
    0 is not included in the output list
    """
    #initialisation
    n = G.number_of_nodes()
    list_of_nodes = list(G.nodes())
    i = 0
    initial_solution = [0]
    
    while len(list_of_nodes) > 1:
        
        list_of_nodes.remove(i) # remove the node which is already in short_path
        k = list_of_nodes[0]
        minimum = G[i][k]['weight']

        # searching for the nearest node
        for j in list_of_nodes:
            if G[i][j]['weight'] < minimum:
                minimum = G[i][j]['weight']
                k = j
        i = k
        initial_solution += [i] # append the nearest node to short_path
    
    initial_solution.remove(0) # remove the initial node
    return initial_solution

  
def random_path(G):
    n = G.number_of_nodes()
    nodes_list = list(range(n))
    random.shuffle(nodes_list)
    
    return nodes_list

  
def adjacency_matrix(path):
    """
    Input: 
    """
    n = len(path)+1
    F = np.zeros((n,n))
    F[0][path[0]] = 1
    F[path[-1]][0] = 1
    for i in range(n-2):
        F[path[i]][path[i+1]] = 1
    return F

  
def tabu_search(G, iterations = 10000, permutations = 10, p = 0.7):
    """
    Input: G is a complete weighted undirected graph
           iterations is the number of times we repeat the algorithm
           permutations is the number of random permutations we generate
           p is the probability of not entering the roulette
    Output: The shortest path, and the shortest distance found by the Tabu search
    """

    #Initialization of variables
    n = G.number_of_nodes()
    tabu = []
    edge_frequency_list = []
    short_path = greedy_path(G)    #initial solution
    short_distance = length(G, short_path)    #initial short distance

    # Choose parameters, should be a function of n
    tabu_list_length = 2 * math.floor(2/5*n) #only change things inside the floor function

    frequency_list_length = math.floor(2/5*n) #only change things inside the floor function

    # Choose penalty base parameter, should be a function of greedy_path(G) and frequency_list_length
    penalty_parameter = length(G, greedy_path(G)) /(20 * frequency_list_length)

    # Choose penalty base parameter, only change things inside the fllor function, should be a function of frequency_list_length
    penalty_limit = math.floor(3/4* frequency_list_length)

    # find the edge list
    A = adjacency_matrix(short_path)
    edge_frequency_list.append(A)

    best_path = short_path
    best_distance = short_distance

    node_pair = [(i, j) for i in range(n-2) for j in range(i+1, n-1)]

    # Beginning the iterations
    for t in range(iterations):

        #Initialization specific for this iteration.
        iteration_node_pair = []
        iteration_edge_frequency_list = []
        iteration_distances = []
        iteration_paths = []
        
        permutation_sample = rand.sample(node_pair, permutations)
        for pair in permutation_sample: 

            # inherit the edge frequency list
            iteration_edge_frequency_list = edge_frequency_list.copy()

            #Generating the random swap.
            i, j = pair

            #Creating a new path even if the elements are in the tabu list, and add to the edge frequency list.
            new_path = short_path.copy()
            new_path[i], new_path[j] = short_path[j], short_path[i]
            A = adjacency_matrix(new_path)
            iteration_edge_frequency_list.append(A)

            #Checking if the frequency list is too big; if it is, remove the first element and add the new one.
            if len(iteration_edge_frequency_list) > frequency_list_length:
                del iteration_edge_frequency_list[0]
            iteration_frequency_matrix = sum([x for x in iteration_edge_frequency_list])
            
            #Computing the weight of the new path.
            #NOTE: Depending on what proves to be efficient, we may change how the frequencies influence the distance.
            #For edges with frequency >= 3, add a penalty of 10 (3, 10 changable)

            new_distance = 0
            new_path = [0] + new_path + [0]

            for k in range(len(new_path)-1):
                new_distance += G[new_path[k]][new_path[k+1]]['weight']
                if iteration_frequency_matrix[new_path[k]][new_path[k+1]] >= penalty_limit:
                    new_distance += penalty_parameter * iteration_frequency_matrix[new_path[k]][new_path[k+1]]
                    """Change parameters later"""
            del new_path[-1]
            del new_path[0]
            
            #Checking for aspiration conditions and completes the iteration-specific lists.
            
            if (new_distance < short_distance and ([short_path[i], short_path[j]] in tabu) and tabu.index([short_path[i], short_path[j]]) < 2) \
               or ([short_path[i], short_path[j]] not in tabu):
                iteration_distances.append(new_distance)
                iteration_node_pair.append([short_path[i],short_path[j]])
                iteration_paths.append(new_path)
            del iteration_edge_frequency_list[-1]
                
        #Obtaining the final results based on the probabilities; if the random number generated is bigger than p,
        #we choose a non-optimal solution; if the random number generated is smaller, we choose the optimal one.
        coin_toss = rand.uniform(0,1)
        if coin_toss < p:
            short_distance = min(iteration_distances)
            index = iteration_distances.index(short_distance)
        # choose a roulette method if not
        else:
            roul = roulette(iteration_distances)
            short_distance = roul[0]
            index = roul[1]
        short_path = iteration_paths[index]
        tabu_pair = iteration_node_pair[index]
        reversed_tabu_pair = (tabu_pair[1], tabu_pair[0])
        tabu.append(tabu_pair)
        tabu.append(reversed_tabu_pair)

        #delete the first entry in tabu list and edge frequency list if they are too long
        if len(tabu) > tabu_list_length: # parameter can be changed
            del tabu[0:2]
        edge_frequency_list.append(adjacency_matrix(short_path))
        if len(edge_frequency_list) > frequency_list_length:
            del edge_frequency_list[0]

        true_distance = length(G,short_path)
        if true_distance < best_distance:
            best_path = short_path
            best_distance = true_distance

    return best_distance, best_path


def exact_solution(G):
        """
        Find the exact optimal solution for graph G.
        """
        n = G.number_of_nodes()
        shortest_distance = length(G, list(range(1, n)))
        for x in itertools.permutations(list(range(1, n))):
            distance = length(G, x)
            if distance < shortest_distance:
                global_min_path = x
                shortest_distance = distance
        return shortest_distance, global_min_path

import numpy as np

def collinear(s0,s1,a = 0.01, b = 0.01, c = 0.01):
    """
    Input: two line segments s0 and s1 with endpoints [A,B], [C,D] respectively
    Output: T/F statement on whether s0, s1 (nearly) cross 
    parameters:
    For graph in [0,1]x[0,1], set a = b = c = 0.01
    These parameters can be change according the range of coordinates
    i.e. a = width_of_G * 0.01
         b = height_of_G * 0.01
         c = width_of_G * 0.01
    """

    if s0[1][0]==s0[0][0]:
        if s1[1][0] == s1[0][0]:
            gradient = [10000,10000]
        else:
            gradient = [10000, (s1[1][1] -s1[0][1])/(s1[1][0] -s1[0][0])]
    else:
        if s1[1][0] == s1[0][0]:
            gradient = [(s0[1][1] -s0[0][1])/(s0[1][0] -s0[0][0]), 10000]
        else:
            gradient = [(s0[1][1] -s0[0][1])/(s0[1][0] -s0[0][0]), (s1[1][1] -s1[0][1])/(s1[1][0] -s1[0][0])]
        
    y_intercept = [s0[0][1] - gradient[0]*s0[0][0], s1[0][1] - gradient[1]*s1[0][0]]
    x_intercept = [-y_intercept[0]/gradient[0], -y_intercept[1]/gradient[1]]

    # check if s0, s1 (nearly) colinear
    #Case 1: s0, s1 is not vertical
    if abs(gradient[0] - gradient[1]) < a and abs(y_intercept[0] - y_intercept[1]) < b:
        return (max(s0[0][0], s0[1][0]) > min(s1[0][0], s1[1][0])) and (min(s0[0][0], s0[1][0]) < max(s1[0][0], s1[1][0]))
    
    #Case 2: s0, s1 is nearly vertical
    if abs(1/gradient[0] - 1/gradient[1]) < a and abs(x_intercept[0] - x_intercept[1]) < c:
        return (max(s0[0][1], s0[1][1]) > min(s1[0][1], s1[1][1])) and (min(s0[0][1], s0[1][1]) < max(s1[0][1], s1[1][1]))
    
    #Case 3: s0, s1 are connected, we only consider the difference between two lines' gradients
    if np.all([np.any([s0[0]==s1[1], s0[1]==s1[0]]), np.any([(abs(gradient[0] - gradient[1])) < a, abs(1/gradient[0] - 1/gradient[1]) < a])]):
        return (max(s0[0][1], s0[1][1]) == max(s1[0][1], s1[1][1])) or (min(s0[0][1], s0[1][1]) == min(s1[0][1], s1[1][1]))


def intersects(s0,s1):

    dx = [s0[1][0]-s0[0][0], s1[1][0]-s1[0][0]]
    dy = [s0[1][1]-s0[0][1], s1[1][1]-s1[0][1]]
    det0 = dy[1]*(s1[1][0]-s0[0][0]) - dx[1]*(s1[1][1]-s0[0][1])
    det1 = dy[1]*(s1[1][0]-s0[1][0]) - dx[1]*(s1[1][1]-s0[1][1])
    det2 = dy[0]*(s0[1][0]-s1[0][0]) - dx[0]*(s0[1][1]-s1[0][1])
    det3 = dy[0]*(s0[1][0]-s1[1][0]) - dx[0]*(s0[1][1]-s1[1][1])

    return det0*det1 < 0 and det2*det3 < 0


def intersection_detection(G, pos, path, a, b, c):
    path = [0] + path + [0]
    new_path = []
    for i in range(len(path)-1):
        for j in range(i+1,len(path)-1):
            s0 = [pos[path[i]], pos[path[i+1]]]
            s1 = [pos[path[j]], pos[path[j+1]]]
            if intersects(s0,s1) == True:

                #make a swap and reverse the order of nodes between i+1 to j
                new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                new_path.remove(0)
                new_path.remove(0)
                return new_path 

            elif collinear(s0,s1,a, b, c) == True:

                #if the swap increases the length, unswap them
                if i+1 == j and j < len(path)-2:
                    diff_l = 0 - G[path[i]][path[i+1]]['weight'] - G[path[j+1]][path[j+2]]['weight']+ G[path[i]][path[j+1]]['weight']+ G[path[j]][path[j+2]]['weight']
                    if diff_l < 0:
                        path[i+1], path[j+1] = path[j+1], path[i+1]
                        path.remove(0)
                        path.remove(0)
                        return path
                    
                else: 
                    diff_l = 0 - G[path[i]][path[i+1]]['weight'] - G[path[j+1]][path[j]]['weight'] + G[path[i]][path[j]]['weight']+G[path[j+1]][path[i+1]]['weight']
                    if diff_l < 0 :
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        new_path.remove(0)
                        new_path.remove(0)
                        return new_path

    path.remove(0)
    path.remove(0)
    return path

def full_intersection_detection(G, pos, path, iterations, a = 0.01, b = 0.01, c = 0.01):
    n = G.number_of_nodes()
    new_path = intersection_detection(G, pos, path,a,b,c)
    i = 1
    while new_path != path and i < iterations:
        path = intersection_detection(G, pos, new_path,a,b,c)
        new_path = intersection_detection(G, pos, path,a,b,c)
        i += 2   
    return new_path, i 

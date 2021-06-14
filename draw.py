import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def draw(G, solution=None, pos=None):
    # draws the solution tour of TSP on graph G. solution is a list with numbers [1,n-1]. pos is a position dictionary of the nodes of graph G.

    if not pos:
        pos = nx.circular_layout(G)  # could also choose other layouts

    nx.draw_networkx_nodes(G, pos, node_color="k", alpha=0.75, node_size=120)  # nx.draw(G, pos) draws all the edges

    node_list = range(G.number_of_nodes())
    node_labels = dict(zip(node_list, [f"{i}" for i in node_list]))
    nx.draw_networkx_labels(G, pos, node_labels, font_color="w", font_size=8)

    if solution:
        solution.append(0)
        solution = [0] + solution
        solution_edges = [(solution[i], solution[i+1]) for i in range(len(solution) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=solution_edges, edge_color='r', width=2)

        weight_list = [G[i[0]][i[1]]['weight'] for i in solution_edges]
        weight_list_string = [str(round(i,2)) for i in weight_list]
        edge_labels = dict(zip(solution_edges, weight_list_string))
        nx.draw_networkx_edge_labels(G, pos, edge_labels, bbox=dict(alpha=0), verticalalignment="baseline", font_size=8)

        ax = plt.subplot(111)
        t = plt.text(0.01, 1.05, f"Path length: {round(sum(weight_list),2)}", transform=ax.transAxes, fontsize=15)
        t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

    plt.show()

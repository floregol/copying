import random
from copy import deepcopy
import numpy as np
from scipy import sparse
import math


def poison_adj_DICE_attack(seed, adj, labels, m, list_test_nodes, percent_corruption_neighbors=0.5):
    random.seed(seed)
    attack_adj = deepcopy(adj).todense()
    nodes_to_corrupt = random.sample(list(list_test_nodes), m)

    for n in nodes_to_corrupt:
        label = np.argmax(labels[n])
        degree = np.sum(attack_adj[n, :])
        num_remove_neighbors = math.ceil(degree * percent_corruption_neighbors)
        list_neighbors = list(np.argwhere(attack_adj[n, :])[:, 1])
        to_zero = random.sample(list_neighbors, num_remove_neighbors)
        attack_adj[n, to_zero] = 0
        attack_adj[to_zero, n] = 0
        new_neighbors = random.sample(range(0, adj.shape[0]), num_remove_neighbors)
        attack_adj[n, new_neighbors] = 1
        attack_adj[new_neighbors, n] = 1

    return sparse.csr_matrix(attack_adj), nodes_to_corrupt

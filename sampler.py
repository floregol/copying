import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from numpy import matlib as mb

def sample_new_pos(m, z, node_index):
    pairwise = rbf_kernel(z)

    np.fill_diagonal(pairwise, 0)
    p = np.power(pairwise[node_index, :], 3) / np.sum(np.power(pairwise[node_index, :], 3)) # will have to try the other distance later
  
    s = np.random.choice(range(0, p.shape[0]), m, p=p)
    
    order = np.argsort(p)
    
    # plt.plot(p[order], 'o')
    # plt.plot(p[order[s]], 'o')
    # plt.show()
    return np.arange(0,z.shape[0], 1)[order][-m:]


def get_new_pos(m, dist, node_index):

    order = np.argsort(dist[node_index])

    return order[1:m+1]


def compute_euclidean_distance(embed):

    N = embed.shape[0]
    p = np.dot(embed, np.transpose(embed))
    q = mb.repmat(np.diag(p), N, 1)
    dist = q + np.transpose(q) - 2 * p
    return dist

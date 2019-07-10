import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


def sample_new_pos(m, z, node_index):
    pairwise = rbf_kernel(z)

    np.fill_diagonal(pairwise, 0)
    p = np.power(pairwise[node_index, :], 3) / np.sum(np.power(pairwise[node_index, :], 3))
  
    s = np.random.choice(range(0, p.shape[0]), m, p=p)
    
    order = np.argsort(p)
    
    # plt.plot(p[order], 'o')
    # plt.plot(p[order[s]], 'o')
    # plt.show()
    return np.arange(0,z.shape[0], 1)[order][-m:]

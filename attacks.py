import random
from copy import deepcopy
import numpy as np
from scipy import sparse
import math
from nettack.GCN import GCN
from nettack.nettack import Nettack
from nettack.utils import preprocess_graph
from fgaattack.fga import FGA
from fgaattack.gcn import GCN2


def poison_adj_NETTACK_attack(seed, adj, labels, features_sparse, m, list_test_nodes, train_mask, val_mask):

    random.seed(seed)
    sizes = [16, labels.shape[1]]
    degrees = adj.A.sum(axis=1).flatten()

    An = preprocess_graph(adj)

    surrogate_model = GCN(sizes, An, features_sparse, with_relu=False, name="surrogate", gpu_id=0)
    split_train = np.argwhere(train_mask).reshape(-1)
    split_val = np.argwhere(val_mask).reshape(-1)
    surrogate_model.train(split_train, split_val, labels)
    W1 = surrogate_model.W1.eval(session=surrogate_model.session)
    W2 = surrogate_model.W2.eval(session=surrogate_model.session)
    nodes_to_corrupt = random.sample(list(list_test_nodes), m)
    direct_attack = True
    perturb_features = False
    perturb_structure = True
    n_influencers = 1
    flatten_labels = np.argwhere(labels)[:, 1].flatten()
    attack_adj = adj
    for u in nodes_to_corrupt:

        nettack = Nettack(attack_adj, features_sparse, flatten_labels, W1, W2, u, verbose=False)
        n_perturbations = int(degrees[u])  # How many perturbations to perform. Default: Degree of the node
        nettack.reset()
        nettack.attack_surrogate(
            n_perturbations,
            perturb_structure=perturb_structure,
            perturb_features=perturb_features,
            direct=direct_attack,
            n_influencers=n_influencers)
        attack_adj = nettack.adj
    return sparse.csr_matrix(attack_adj), nodes_to_corrupt


def poison_adj_DICE_attack(seed, adj, labels, communities, m, list_test_nodes, percent_corruption_neighbors):
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
        new_neighbors = random.sample(list(np.setdiff1d(range(adj.shape[0]), communities[label])), num_remove_neighbors)
        attack_adj[n, new_neighbors] = 1
        attack_adj[new_neighbors, n] = 1

    return sparse.csr_matrix(attack_adj), nodes_to_corrupt


def poison_adj_DISCONNECTING_attack(seed, adj, m, list_test_nodes):
    random.seed(seed)
    attack_adj = deepcopy(adj)
    nodes_to_corrupt = random.sample(list(list_test_nodes), m)

    for n in nodes_to_corrupt:

        attack_adj[n, :] = 0
        attack_adj[:, n] = 0

    return sparse.csr_matrix(attack_adj), nodes_to_corrupt


def poison_adj_fga_attack(seed, adj, labels, features_sparse, m, list_test_nodes, train_mask, val_mask):

    random.seed(seed)
    sizes = [16, labels.shape[1]]
    degrees = adj.A.sum(axis=1).flatten()

    An = preprocess_graph(adj)
    flatten_labels = np.argwhere(labels)[:, 1].flatten()
    device = 'cpu'
    surrogate_model = GCN2(
        nfeat=features_sparse.shape[1], nclass=flatten_labels.max().item() + 1, nhid=16, with_relu=False, device=device)
    surrogate_model = surrogate_model.to(device)
    split_train = np.argwhere(train_mask).reshape(-1)
    split_val = np.argwhere(val_mask).reshape(-1)
    surrogate_model.fit(features_sparse, An, flatten_labels, split_train)

    nodes_to_corrupt = random.sample(list(list_test_nodes), m)

    attack_adj = adj
    for u in nodes_to_corrupt:

        n_perturbations = int(degrees[u])  # How many perturbations to perform. Default: Degree of the node
        fga = FGA(model=surrogate_model, nnodes=adj.shape[0], feature_shape=features_sparse.shape, device=device)
        fga = fga.to(device)
        fga.attack(
            features_sparse, attack_adj, flatten_labels, split_train, target_node=0, n_perturbations=n_perturbations)
        attack_adj = fga.modified_adj
    return sparse.csr_matrix(attack_adj), nodes_to_corrupt
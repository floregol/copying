import numpy as np
from defense import GCN, GCNSVD
import torch
from attacks import *
from helper import *
from gvae.GVAE import get_z_embedding
from sampler import *
from utils import preprocess_adj, sparse_to_tuple
from copy import copy, deepcopy
from gcn.train_gcn import get_trained_gcn
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
import scipy as sp
import time


def run_trial(train_index, test_index, seed, trial, labels, adj, features_sparse, feature_matrix, initial_num_labels,
              attack_name, percent_corruption_neighbors, num_attacked_nodes, new_positions):
    n = feature_matrix.shape[0]
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
                                                                        initial_num_labels)
    idx_train = np.argwhere(train_mask).reshape(-1)
    ground_truth = np.argmax(labels, axis=1)
    K = np.max(ground_truth) + 1
    communities = {}
    for k in range(K):
        communities[k] = np.where(ground_truth == k)[0]
    flatten_labels = np.argwhere(labels)[:, 1].flatten()

    number_labels = labels.shape[1]
    np.random.seed(seed)  # set the seed for current trial
    random.seed(seed)
    print('========================================================================================================')
    print('trial : ' + str(trial))
    """
    First, corrupt the adjacency matrix for a a fixed number of randomly sleected nodes from the test set.
    """
    if attack_name == 'dice':

        attacked_adj, attacked_nodes = poison_adj_DICE_attack(seed, adj, labels, communities, num_attacked_nodes,
                                                              test_index, percent_corruption_neighbors)

    elif attack_name == 'nettack':
        attacked_adj, attacked_nodes = poison_adj_NETTACK_attack(seed, adj, labels, features_sparse, num_attacked_nodes,
                                                                 test_index, train_mask, val_mask)
    else:
        attacked_adj, attacked_nodes = poison_adj_DISCONNECTING_attack(seed, adj, num_attacked_nodes, test_index)

    full_A_tilde = preprocess_adj(attacked_adj, dense=False, spr=True)
    """
    Train a GCN to get a predictor to evaluate the accuracy at the attacked nodes.
    """
    w_0, w_1, A_tilde, gcn_soft, close = get_trained_gcn(seed, attacked_adj, features_sparse, y_train, y_val, y_test,
                                                         train_mask, val_mask, test_mask)

    # Get prediction by the GCN
    initial_gcn = gcn_soft(sparse_to_tuple(features_sparse))

    full_pred_gcn = np.argmax(initial_gcn, axis=1)

    gcn_softmax = deepcopy(initial_gcn)
    new_pred_no_copy = deepcopy(full_pred_gcn)
    new_pred_majority_no_copy = deepcopy(full_pred_gcn)

    new_pred = deepcopy(full_pred_gcn)
    new_pred_majority = deepcopy(full_pred_gcn)

    print("ACC old pred at attacked nodes: " +
          str(accuracy_score(ground_truth[attacked_nodes], full_pred_gcn[attacked_nodes])))

    # model = GCNSVD(nfeat=features_sparse.shape[1], nclass=K, nhid=16, device='cpu')
    # model = model.to('cpu')
    # model.fit(features_sparse, attacked_adj, flatten_labels, idx_train)
    # gcnsvd_acc = model.test(attacked_nodes).item()
    gcnsvd_acc = 0
    """
    Get the embeddings of all the nodes
    """
    z = get_z_embedding(attacked_adj, features_sparse, labels, seed, verbose=False)

    dist = compute_euclidean_distance(z)

    j = 0
    """
    Now or each attacked nodes, we copy it to new positions hoping tha we can recover the true label
    """
    for node_index in attacked_nodes:

        node_features = deepcopy(feature_matrix[node_index])
        start_time = time.time()

        node_true_label = ground_truth[node_index]
        node_thinking_label = full_pred_gcn[node_index]

        list_new_posititons = get_new_pos(new_positions, dist, node_index)

        def move_node(list_new_posititons, feature_matrix, number_labels, full_A_tilde, w_0, w_1, node_features):
            i = 0
            softmax_output_list = np.zeros((new_positions, number_labels))
            for new_spot in list_new_posititons:
                replaced_node_label = ground_truth[new_spot]
                saved_features = deepcopy(
                    feature_matrix[new_spot])  # save replaced node features to do everything in place (memory)

                # move the node to the new position
                feature_matrix[new_spot] = node_features
                start = time.time()
                softmax_output_of_node = fast_localized_softmax(feature_matrix, new_spot, full_A_tilde, w_0,
                                                                w_1)  # get new softmax output at this position
                end = time.time()
                # print(end-start)

                obt_label = np.argmax(softmax_output_of_node)

                softmax_output_list[i] = softmax_output_of_node
                i += 1

                # undo changes on the feature matrix
                feature_matrix[new_spot] = saved_features
                obtained_labels_list = np.argmax(softmax_output_list, axis=1)
            return softmax_output_list, obtained_labels_list

        # To store results
        softmax_output_list, obtained_labels_list = move_node(list_new_posititons, features_sparse, number_labels,
                                                              full_A_tilde, w_0, w_1, node_features)
        """
        compute new label by averaging without copying
        """
        softmax_output_list_no_copy = gcn_softmax[list_new_posititons]
        y_bar_x_no_copy = np.mean(softmax_output_list_no_copy, axis=0)
        obtained_labels_list_no_copy = np.argmax(softmax_output_list_no_copy, axis=1)

        new_label_no_copy = np.argmax(y_bar_x_no_copy, axis=0)
        new_label_majority_no_copy = sp.stats.mode(obtained_labels_list_no_copy)[0]

        new_pred_no_copy[node_index] = new_label_no_copy
        new_pred_majority_no_copy[node_index] = new_label_majority_no_copy
        """
        compute new label by averaging after copying
        """
        y_bar_x = np.mean(softmax_output_list, axis=0)
        new_label = np.argmax(y_bar_x, axis=0)
        new_label_majority = sp.stats.mode(obtained_labels_list)[0]

        new_pred[node_index] = new_label
        new_pred_majority[node_index] = new_label_majority

        j += 1

    close()

    gcn_acc = accuracy_score(ground_truth[attacked_nodes], full_pred_gcn[attacked_nodes])

    nocopy_acc = accuracy_score(ground_truth[attacked_nodes], new_pred_no_copy[attacked_nodes])
    nocopy_maj_acc = accuracy_score(ground_truth[attacked_nodes], new_pred_majority_no_copy[attacked_nodes])

    copy_acc = accuracy_score(ground_truth[attacked_nodes], new_pred[attacked_nodes])
    copy_maj_acc = accuracy_score(ground_truth[attacked_nodes], new_pred_majority[attacked_nodes])

    print("ACC old pred : ", gcn_acc)
    print("ACC baseline at attacked nodes", gcnsvd_acc)
    print("ACC new pred (copying) : ", copy_acc)
    print()
    print("ACC new pred (no copying) (avg. softmax) : ", nocopy_acc)
    print("ACC new pred (no copying) (majority vote) : ", nocopy_maj_acc)
    print("ACC new pred (copying) (majority vote) : ", copy_maj_acc)

    return gcn_acc, gcnsvd_acc, copy_acc, copy_maj_acc, nocopy_acc, nocopy_maj_acc

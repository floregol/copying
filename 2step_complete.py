import time
from utils import load_data, preprocess_adj, preprocess_features, sparse_to_tuple
import numpy as np
# import os
# from scipy import sparse
from gcn.train_gcn import get_trained_gcn
from copy import copy, deepcopy
# import pickle as pk
# import multiprocessing as mp
# import math
# import sys
from sklearn.metrics import accuracy_score
from sampler import sample_new_pos
# from scipy.stats import entropy
# import math
# import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from helper import *
"""

 Moving the nodes around experiment

"""
NUM_CROSS_VAL = 4
trials = 2
CORES = 4
# Train the GCN
SEED = 43
QUANTILE = 0.75
initial_num_labels = 20
m = 5
dataset = 'cora'
adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)
ground_truth = np.argmax(labels, axis=1)
A = adj.todense()
full_A_tilde = preprocess_adj(adj, True)
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
n = feature_matrix.shape[0]
number_labels = labels.shape[1]

test_split = StratifiedShuffleSplit(n_splits=NUM_CROSS_VAL, test_size=0.37, random_state=SEED)
test_split.get_n_splits(labels, labels)
seed_list = [1, 2, 3, 4]

for train_index, test_index in test_split.split(labels, labels):

    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
                                                                        initial_num_labels)

    for trial in range(trials):
        seed = seed_list[trial]
        w_0, w_1, A_tilde, gcn_soft = get_trained_gcn(seed, dataset, y_train, y_val, y_test, train_mask, val_mask,
                                                      test_mask)

        # Get prediction by the GCN
        initial_gcn = gcn_soft(sparse_to_tuple(features_sparse))

        full_pred_gcn = np.argmax(initial_gcn, axis=1)

        print("ACC old pred : " + str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))
        log_odds_ratio_gcn = np.apply_along_axis(log_odds_ratio, 1, initial_gcn)

        score = np.array(log_odds_ratio_gcn[test_index])
       
        threshold = np.quantile(score, QUANTILE)
        
        test_nodes_to_reclassify = test_index[np.argwhere(score < threshold)]
        scores_reclassify = score[np.argwhere(score < threshold)]
        j = 0
        for node_index in test_nodes_to_reclassify:  # TODO in parrallel copy features matrix

            node_features = deepcopy(feature_matrix[node_index])
            start_time = time.time()

            node_true_label = ground_truth[node_index]
            node_thinking_label = full_pred_gcn[node_index]

            list_new_posititons = sample_new_pos(m,node_features)
            def move_node(list_new_posititons, feature_matrix, softmax_output_list, number_labels, full_A_tilde, w_0,
                          w_1, node_features):
                i = 0
                softmax_output_list = np.zeros((len(list_new_posititons), number_labels))
                for new_spot in list_new_posititons:
                    saved_features = deepcopy(
                        feature_matrix[new_spot])  # save replaced node features to do everything in place (memory)

                    feature_matrix[new_spot] = node_features  # move the node to the new position

                    softmax_output_of_node = fast_localized_softmax(feature_matrix, new_spot, full_A_tilde, w_0,
                                                                    w_1)  # get new softmax output at this position

                    softmax_output_list[i] = softmax_output_of_node  # Store results
                    i += 1
                    # print("put at " + str(replaced_node_label) + " = " + str(np.argmax(softmax_output_of_node)))

                    feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix
                return softmax_output_list

            #To store results
            softmax_output_list = np.zeros((len(list_new_posititons), number_labels))
            partition_size = int(len(list_new_posititons) / CORES)
            start_index = list(range(0, len(list_new_posititons), partition_size))
            end_index = [i for i in start_index[1:]]
            end_index.append(len(list_new_posititons))
            splited_list = [list(list_new_posititons[start_index[i]:end_index[i]]) for i in range(CORES)]

            softmax_output_lists = [np.zeros((len(i), number_labels)) for i in splited_list]
            pool = mp.Pool(processes=CORES)
            pool_results = [
                pool.apply_async(move_node, (splited_list[i], feature_matrix, softmax_output_lists[i], number_labels,
                                             full_A_tilde, w_0, w_1, node_features)) for i in range(CORES)
            ]
            pool.close()
            pool.join()
            i_results = 0

            for pr in pool_results:
                thread_results = pr.get()
                softmax_output_list[start_index[i_results]:end_index[i_results]] = thread_results

            y_bar_x = np.mean(softmax_output_list, axis=0)
            print(j)
            new_label = np.argmax(y_bar_x, axis=0)
            neighbors_labels = full_pred_gcn[np.argwhere(A[node_index])[:, 1]]
            similar_neighbors = np.where(neighbors_labels == new_label)[0].shape[0]
            num_neighbors = neighbors_labels.shape[0]

            if similar_neighbors / num_neighbors > scores_reclassify[j]:
                new_pred_soft[node_index] = new_label
            # print(str(node_true_label) + " pred " + str(node_thinking_label) + " new : " + str(new_label))

            y_bar_x = y_bar_x - initial_avergae

            new_label = np.argmax(y_bar_x, axis=0)
            neighbors_labels = full_pred_gcn[np.argwhere(A[node_index])[:, 1]]
            similar_neighbors = np.where(neighbors_labels == new_label)[0].shape[0]
            num_neighbors = neighbors_labels.shape[0]

            if similar_neighbors / num_neighbors > scores_reclassify[j]:
                new_pred_wei_soft[node_index] = new_label

            j += 1

        print("ACC old pred : " + str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))

        print("ACC soft  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_soft[test_index])))

        print("ACC corrected  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_wei_soft[test_index])))

        # print("ACC log neigh  pred : " +
        #       str(accuracy_score(ground_truth[test_index], new_pred_log_neigh_wei[test_index])))

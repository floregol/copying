import pickle as pk
from attacks import *
from helper import *
from sklearn.model_selection import StratifiedShuffleSplit
from gvae.GVAE import get_z_embedding
from sampler import *
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from copy import copy, deepcopy
from gcn.train_gcn import get_trained_gcn
import scipy.stats
import scipy as sp
import numpy as np
from utils import load_data, preprocess_adj, preprocess_features, sparse_to_tuple
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import multiprocess as mp
"""

 Moving the nodes around experiment

"""

trials = 3
dataset = 'cora'
attack_name = 'nettack'  # nettack
percent_corruption_neighbors = 0.75

num_attacked_nodes = 50
new_positions = 10

initial_num_labels = 10
SEED = 123
np.random.seed(SEED)

NUM_CROSS_VAL = 3
#CORES = 4

print('========================================================================================================')
print('The datset is : ' + str(dataset))
print('Number of attacked nodes : ' + str(num_attacked_nodes))
print('Number of nodes for copying at each of attacked nodes : ' + str(new_positions))
print('Dice attack is : ' + attack_name)
if attack_name == 'nettack':
    print('Percentage of neighbours corrupted : ' + str(100 * percent_corruption_neighbors))
elif attack_name == 'nettack':
    print('nettack')
else:
    print('No specific attack, the attacked nodes are entirely disconnected')

print('Number of trials : ' + str(trials))

adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)

ground_truth = np.argmax(labels, axis=1)

K = np.max(ground_truth) + 1
communities = {}
for k in range(K):
    communities[k] = np.where(ground_truth == k)[0]

features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
n = feature_matrix.shape[0]
number_labels = labels.shape[1]

test_split = StratifiedShuffleSplit(n_splits=NUM_CROSS_VAL, test_size=0.37, random_state=SEED)
test_split.get_n_splits(labels, labels)
seed_list = np.random.randint(1, 1e6, trials)

for train_index, test_index in test_split.split(labels, labels):

    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
                                                                        initial_num_labels)

    N = len(y_train)

    acc_old_list = np.zeros(trials)

    acc_new_no_copy_list = np.zeros(trials)
    acc_new_no_copy_majority_list = np.zeros(trials)

    acc_new_list = np.zeros(trials)
    acc_new_majority_list = np.zeros(trials)

    for trial in range(trials):
        seed = seed_list[trial]

        np.random.seed(seed)  # set the seed for current trial
        random.seed(seed)
        print(
            '========================================================================================================')
        print('trial : ' + str(trial))
        """
        First, corrupt the adjacency matrix for a a fixed number of randomly sleected nodes from the test set.
        """
        if attack_name == 'DICE':
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
        w_0, w_1, A_tilde, gcn_soft, close = get_trained_gcn(seed, attacked_adj, features_sparse, y_train, y_val,
                                                             y_test, train_mask, val_mask, test_mask)

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
        """
        Get the embeddings of all the nodes
        """
        z = get_z_embedding(attacked_adj, features_sparse, labels, seed, verbose=False)

        dist = compute_euclidean_distance(z)

        j = 0
        """
        Now or each attacked nodes, we copy it to new positions hoping tha we can recover the true label
        """
        for node_index in attacked_nodes:  # TODO in parrallel copy features matrix

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

        acc_old_list[trial] = accuracy_score(ground_truth[attacked_nodes], full_pred_gcn[attacked_nodes])

        acc_new_no_copy_list[trial] = accuracy_score(ground_truth[attacked_nodes], new_pred_no_copy[attacked_nodes])
        acc_new_no_copy_majority_list[trial] = accuracy_score(ground_truth[attacked_nodes],
                                                              new_pred_majority_no_copy[attacked_nodes])

        acc_new_list[trial] = accuracy_score(ground_truth[attacked_nodes], new_pred[attacked_nodes])
        acc_new_majority_list[trial] = accuracy_score(ground_truth[attacked_nodes], new_pred_majority[attacked_nodes])

        print("ACC old pred : " + str(acc_old_list[trial]))

        print("ACC new pred (no copying) (avg. softmax) : " + str(acc_new_no_copy_list[trial]))
        print("ACC new pred (no copying) (majority vote) : " + str(acc_new_no_copy_majority_list[trial]))

        print("ACC new pred (copying) (avg. softmax) : " + str(acc_new_list[trial]))
        print("ACC new pred (copying) (majority vote) : " + str(acc_new_majority_list[trial]))

print('========================================================================================================')
print('Mean and std. error of GCN accuracy at attacked nodes : {} and {}'.format(
    np.mean(acc_old_list) * 100,
    np.std(acc_old_list) * 100))

print('========================================================================================================')

print('Mean and std. error of accuracy at attacked nodes (no copying) (avg. softmax) : {} and {}'.format(
    np.mean(acc_new_no_copy_list) * 100,
    np.std(acc_new_no_copy_list) * 100))
print('Mean and std. error of accuracy at attacked nodes (no copying) (majority vote) : {} and {}'.format(
    np.mean(acc_new_no_copy_majority_list) * 100,
    np.std(acc_new_no_copy_majority_list) * 100))

print('========================================================================================================')

print('Mean and std. error of Copying model accuracy at attacked nodes  (avg. softmax) : {} and {}'.format(
    np.mean(acc_new_list) * 100,
    np.std(acc_new_list) * 100))
print('Mean and std. error of Copying model accuracy at attacked nodes  (majority vote) : {} and {}'.format(
    np.mean(acc_new_majority_list) * 100,
    np.std(acc_new_majority_list) * 100))

print('========================================================================================================')

_, p_value_no_copy = sp.stats.wilcoxon(acc_old_list, acc_new_no_copy_list, zero_method='wilcox', correction=False)
print('The p value from Wilcoxon signed rank test (no copying) (avg. softmax): ' + str(p_value_no_copy))

_, p_value_no_copy_majority = sp.stats.wilcoxon(
    acc_old_list, acc_new_no_copy_majority_list, zero_method='wilcox', correction=False)
print('The p value from Wilcoxon signed rank test (no copying) (majority vote): ' + str(p_value_no_copy_majority))

print('========================================================================================================')

_, p_value = sp.stats.wilcoxon(acc_old_list, acc_new_list, zero_method='wilcox', correction=False)
print('The p value from Wilcoxon signed rank test (copying) (avg. softmax): ' + str(p_value))

_, p_value_majority = sp.stats.wilcoxon(acc_old_list, acc_new_majority_list, zero_method='wilcox', correction=False)
print('The p value from Wilcoxon signed rank test (copying) (majority vote): ' + str(p_value_majority))

to_store = {
    'acc_old_list': acc_old_list,
    'acc_new_no_copy_list': acc_new_no_copy_list,
    'acc_new_no_copy_majority_list': acc_new_no_copy_majority_list,
    'acc_new_list': acc_new_list,
    'acc_new_majority_list': acc_new_majority_list
}


filename = str(trials) + 'trials' + '_' + dataset + 'dataset' + '_' + str(percent_corruption_neighbors) + \
    'percent_corruption_neighbors' + '_' + \
    str(initial_num_labels) + 'initial_num_labels.pk'
with open(os.path.join('results', filename), 'wb') as f:
    pk.dump(to_store, f)
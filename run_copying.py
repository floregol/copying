import time
from utils import load_data, preprocess_adj, preprocess_features, sparse_to_tuple
import numpy as np
from gcn.train_gcn import get_trained_gcn
from copy import copy, deepcopy
from sklearn.metrics import accuracy_score
from sampler import sample_new_pos
from gvae.GVAE import get_z_embedding
from sklearn.model_selection import StratifiedShuffleSplit
from helper import *
from attacks import poison_adj_DICE_attack
# import multiprocess as mp
"""

 Moving the nodes around experiment

"""
NUM_CROSS_VAL = 1
trials = 2
#CORES = 4
SEED = 123
np.random.seed(SEED)
initial_num_labels = 20
new_positions = 10
num_attacked_nodes = 50
dataset = 'cora'
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

    acc_old_list = np.zeros(trials)
    acc_new_list = np.zeros(trials)

    for trial in range(trials):
        seed = seed_list[trial]

        np.random.seed(seed)  # set the seed for current trial
        random.seed(seed)

        print('trial : ' + str(trial))

        """
        First, corrupt the adjacency matrix for a a fixed number of randomly sleected nodes from the test set.
        """
        attacked_adj, attacked_nodes = poison_adj_DICE_attack(seed, adj, labels, communities, num_attacked_nodes, test_index)

        full_A_tilde = preprocess_adj(attacked_adj, True)

        """
        Train a GCN to get a predictor to evaluate the accuracy at the attacked nodes.
        """
        w_0, w_1, A_tilde, gcn_soft, close = get_trained_gcn(seed, attacked_adj, features_sparse, y_train, y_val,
                                                             y_test, train_mask, val_mask, test_mask)

        # Get prediction by the GCN
        initial_gcn = gcn_soft(sparse_to_tuple(features_sparse)) # could you explain this?
        
        full_pred_gcn = np.argmax(initial_gcn, axis=1)
        new_pred = deepcopy(full_pred_gcn)
        print("ACC old pred at attacked nodes: " +
              str(accuracy_score(ground_truth[attacked_nodes], full_pred_gcn[attacked_nodes])))


        """
        Get the embeddings of all the nodes
        """
        z = get_z_embedding(attacked_adj, features_sparse, labels, seed, verbose=False)
        j = 0

        """
        Now or each attacked nodes, we copy it to new positions hoping tha we can recover the true label
        """
        for node_index in attacked_nodes:  # TODO in parrallel copy features matrix

            node_features = deepcopy(feature_matrix[node_index])
            start_time = time.time()

            node_true_label = ground_truth[node_index]
            node_thinking_label = full_pred_gcn[node_index]

            list_new_posititons = sample_new_pos(new_positions, z, node_index)

            # print('node_true_label ' + str(node_true_label))
            # print('node_thinking_label ' + str(node_thinking_label))
            # print('list_new_posititons ' + str(ground_truth[list_new_posititons]))


            def move_node(list_new_posititons, feature_matrix, number_labels, full_A_tilde, w_0, w_1, node_features):
                i = 0
                softmax_output_list = np.zeros((len(list_new_posititons), number_labels))
                for new_spot in list_new_posititons:
                    replaced_node_label = ground_truth[new_spot]
                    saved_features = deepcopy(
                        feature_matrix[new_spot])  # save replaced node features to do everything in place (memory)

                    feature_matrix[new_spot] = node_features  # move the node to the new position

                    softmax_output_of_node = fast_localized_softmax(feature_matrix, new_spot, full_A_tilde, w_0,
                                                                    w_1)  # get new softmax output at this position

                    l = np.argmax(softmax_output_of_node)

                    softmax_output_list[i] = softmax_output_of_node
                    i += 1

                    feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix
                return softmax_output_list

            #To store results
            softmax_output_list = move_node(list_new_posititons, feature_matrix, number_labels, full_A_tilde, w_0, w_1,
                                            node_features)



            # partition_size = int(len(list_new_posititons) / CORES)

            # start_index = list(range(0, len(list_new_posititons), partition_size))
            # end_index = [i for i in start_index[1:]]
            # end_index.append(len(list_new_posititons))
            # splited_list = [list(list_new_posititons[start_index[i]:end_index[i]]) for i in range(CORES)]

            # softmax_output_lists = [np.zeros((len(i), number_labels)) for i in splited_list]
            # pool = mp.Pool(processes=CORES)
            # pool_results = [
            #     pool.apply_async(move_node, (splited_list[i], feature_matrix, softmax_output_lists[i], number_labels,
            #                                  full_A_tilde, w_0, w_1, node_features)) for i in range(CORES)
            # ]
            # pool.close()
            # pool.join()
            # i_results = 0

            # for pr in pool_results:
            #     thread_results = pr.get()
            #     softmax_output_list[start_index[i_results]:end_index[i_results]] = thread_results
            #     i_results += 1
            #print(softmax_output_list)

            """
            compute new label by averaging after copying
            """
            y_bar_x = np.mean(softmax_output_list, axis=0)
            new_label = np.argmax(y_bar_x, axis=0)
           
           
            # print(new_label)
            # print(new_label)
            #  print(str(node_true_label) + " pred " + str(node_thinking_label) + " new : " + str(new_label))

            new_pred[node_index] = new_label

            j += 1
        close()

        acc_old_list[trial] = accuracy_score(ground_truth[attacked_nodes], full_pred_gcn[attacked_nodes])
        acc_new_list[trial] = accuracy_score(ground_truth[attacked_nodes], new_pred[attacked_nodes])

        print("ACC old pred : " + str(acc_old_list[trial]))

        print("ACC new pred : " + str(acc_new_list[trial]))


print('Mean and std. error of GCN accuracy at attacked nodes : {} and {}'.format(np.mean(acc_old_list)*100, np.std(acc_old_list)*100))
print('Mean and std. error of Copying model accuracy at attacked nodes : {} and {}'.format(np.mean(acc_new_list)*100, np.std(acc_new_list)*100))

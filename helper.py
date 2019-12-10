import random
import numpy as np


def get_labeled_train_index(train_index, labels, num_label):
    labeled_train_index = []
    for l in range(labels.shape[1]):
        training_label_index = np.where(np.argwhere(labels[train_index])[:, 1] == l)[0]
        index_to_add = training_label_index[0:num_label]
        labeled_train_index = labeled_train_index + list(index_to_add)
    return labeled_train_index


def get_split(n, train_index, test_index, labels, initial_num_labels):
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    labeled_train_index = get_labeled_train_index(train_index[0:1208], labels, initial_num_labels)

    train_mask[labeled_train_index] = True
    val_mask[train_index[1208:]] = True
    test_mask[test_index] = True

    y_train = np.zeros(labels.shape, dtype=int)
    y_val = np.zeros(labels.shape, dtype=int)
    y_test = np.zeros(labels.shape, dtype=int)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def log_odds_ratio(softmax_vector):
    sorted_softmax_arg = softmax_vector[np.argsort(softmax_vector)]
    p_max = sorted_softmax_arg[-1]
    p_second_max = sorted_softmax_arg[-2]
    return np.log((p_max * (1 - p_second_max)) / ((1 - p_max) * p_second_max))


def score_percent_similar(list_nodes, full_pred_gcn, A):
    percent_predicted_similar = []
    for i in list_nodes:
        pred_lab = full_pred_gcn[i]
        neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
        similar_neighbors = np.where(neighbors_labels == pred_lab)[0].shape[0]
        num_neighbors = neighbors_labels.shape[0]
        percent_predicted_similar.append(similar_neighbors / num_neighbors)

    return percent_predicted_similar


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def fast_localized_softmax(features, new_spot, full_A_tilde, w_0, w_1):
    neighbors_index = np.argwhere(full_A_tilde[new_spot, :])[:, 1]
    A_neig = full_A_tilde[neighbors_index, :]
  
    H_out = A_neig @ features @ w_0
   # H_out = H_out
    relu_out = np.maximum(H_out, 0, H_out)
    A_i_neigh = full_A_tilde[new_spot, neighbors_index]
    H2_out = A_i_neigh @ relu_out @  w_1
    return softmax(H2_out)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle as pk
import scipy.stats
import scipy as sp
import numpy as np
from utils import load_data, preprocess_features
from sklearn.model_selection import StratifiedShuffleSplit
import time
import sys
from defense_trial import run_trial

result_path = 'results'

num_attacked_nodes = 50
new_positions = 10
SEED = 123
np.random.seed(SEED)
"""
Parse input argument
"""
# python run_copying.py cora 50 10 dice 0.5
dataset = sys.argv[1]
trials = int(sys.argv[2])  # num of random partition
num_cores = int(sys.argv[3])
attack_name = sys.argv[4]  # nettack
initial_num_labels = int(sys.argv[5])
percent_corruption_neighbors = 0
if attack_name == 'dice':  # get the beta percent
    percent_corruption_neighbors = float(sys.argv[6])

print('========================================================================================================')
print('The datset is : ' + str(dataset))
print('Number of attacked nodes : ' + str(num_attacked_nodes))
print('Number of nodes for copying at each of attacked nodes : ' + str(new_positions))
print('Attack is : ' + attack_name)
if attack_name == 'dice':
    print('Percentage of neighbours corrupted : ' + str(100 * percent_corruption_neighbors))
elif attack_name == 'nettack':
    print('nettack')
elif attack_name == 'fga':
    print('fga')
else:
    print('No specific attack, the attacked nodes are entirely disconnected')

print('Number of trials : ' + str(trials))

adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)

features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()


test_split = StratifiedShuffleSplit(n_splits=trials, test_size=0.5, random_state=SEED)
test_split.get_n_splits(labels, labels)
seed_list = np.random.randint(1, 1e6, trials)

acc_old_list = []
acc_new_no_copy_list = []
acc_baseline_gcnsvd_list = []
acc_new_no_copy_majority_list = []
acc_new_list = []
acc_new_majority_list = []

trial = 0
for train_index, test_index in test_split.split(labels, labels):
    gcn_acc, gcnsvd_acc, copy_acc, copy_maj_acc, nocopy_acc, nocopy_maj_acc = run_trial(
        train_index, test_index, seed_list[trial], trial, labels, adj, features_sparse, feature_matrix,
        initial_num_labels, attack_name, percent_corruption_neighbors, num_attacked_nodes, new_positions)
    trial += 1

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

print('Mean and std. error of Baseline GCNSVD accuracy at attacked nodes  : {} and {}'.format(
    np.mean(acc_baseline_gcnsvd_list) * 100,
    np.std(acc_baseline_gcnsvd_list) * 100))

print('========================================================================================================')

_, p_value_majority = sp.stats.wilcoxon(acc_old_list, acc_new_list, zero_method='wilcox', correction=False)
print('The p value from Wilcoxon signed rank test (copying) (gcn): ' + str(p_value_majority))

_, p_value_svd = sp.stats.wilcoxon(acc_baseline_gcnsvd_list, acc_new_list, zero_method='wilcox', correction=False)
print('The p value from Wilcoxon signed rank test (copying) (svd): ' + str(p_value_svd))

to_store = {
    'acc_old_list': acc_old_list,
    'acc_new_no_copy_list': acc_new_no_copy_list,
    'acc_new_no_copy_majority_list': acc_new_no_copy_majority_list,
    'acc_new_list': acc_new_list,
    'acc_new_majority_list': acc_new_majority_list,
    'acc_baseline_gcnsvd_list': acc_baseline_gcnsvd_list
}


filename = str(trials) + '_trials' + '_' + dataset + 'dataset' + '_' + str(percent_corruption_neighbors) + \
    '_percent_corruption_neighbors' + '_' + \
    str(initial_num_labels) + '_initial_num_labels.pk'

if not os.path.exists(result_path):
    os.makedirs(result_path)

with open(os.path.join(result_path, filename), 'wb') as f:
    pk.dump(to_store, f)
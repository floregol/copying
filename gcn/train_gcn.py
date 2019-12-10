from __future__ import division
from __future__ import print_function

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from utils import *
from gcn.models import GCN, MLP
import os
from scipy import sparse

QUICK_MODE = False

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
if QUICK_MODE:
    flags.DEFINE_integer('epochs', 3, 'Number of epochs to train.')
else:
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def get_trained_gcn(seed, adj, features_sparse, y_train, y_val, y_test, train_mask, val_mask, test_mask, mlp=False, VERBOSE=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'
    # Set random seed
    seed = seed
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Some preprocessing
    features = sparse_to_tuple(features_sparse)

    support = [preprocess_adj(adj)]
    num_supports = 1
    if mlp:
        model_func = MLP
    else:
        model_func = GCN


    # Define placeholders
    placeholders = {
        'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.compat.v1.placeholder(tf.int32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
            tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    input_dim = features[2][1]
    # Create model
    model = model_func(placeholders, input_dim=input_dim, logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Added this function to obtain softmax output
    def softmax(features):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        all_softamx_output, _, _ = sess.run([model.softmax_ouput, model.loss, model.accuracy], feed_dict=feed_dict_val)
        return all_softamx_output

    def close_session():
        sess.close()
        tf.reset_default_graph()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    # print("Training the GCN with Kipf values....")
    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        if VERBOSE:
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=",
                  "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),
                  "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    # print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "time=",
          "{:.5f}".format(test_duration))

    for w in tf.trainable_variables():

        if w.shape[0] == input_dim:
            w_0 = sess.run(model.vars[w.name])
        else:
            w_1 = sess.run(model.vars[w.name])

    A_tilde = support

    return w_0, w_1, A_tilde, softmax, close_session

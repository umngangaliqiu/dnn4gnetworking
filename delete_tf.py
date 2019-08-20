# A simple Tensorflow 2 layer dense network example
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the sklearn breast cancer dataset
bc = datasets.load_breast_cancer()
X = bc.data[:, :]
Y = bc.target

# min max scale and binarize the target labels
scaler = MinMaxScaler()
X = scaler.fit_transform(X,Y)
label = LabelBinarizer()
Y = label.fit_transform(Y)

# train fraction
frac = 0.9

# shuffle dataset
idx = np.random.randint(X.shape[0], size=len(X))
X = X[idx]
Y = Y[idx]

train_stop = int(len(X) * frac)

X_ = X[:train_stop]
Y_ = Y[:train_stop]

X_t = X[train_stop:]
Y_t = Y[train_stop:]

# plot the first 3 PCA dimensions of the sampled data
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X_)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y_.ravel(),
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

# create the TF neural net
# some hyperparams
training_epochs = 200

n_neurons_in_h1 = 10
n_neurons_in_h2 = 10
learning_rate = 0.1

n_features = len(X[0])
labels_dim = 1
#############################################

# basic 2 layer dense net (MLP) example adapted from
# https://becominghuman.ai/creating-your-own-neural-network-using-tensorflow-fa8ca7cc4d0e

# these placeholders serve as our input tensors
x = tf.placeholder(tf.float32, [None, n_features], name='input')
y = tf.placeholder(tf.float32, [None, labels_dim], name='labels')

# TF Variables are our neural net parameter tensors, we initialize them to random (gaussian) values in
# Layer1. Variables are allowed to be persistent across training epochs and updatable bt TF operations
W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')

# note the output tensor of the 1st layer is the activation applied to a
# linear transform of the layer 1 parameter tensors
# the matmul operation calculates the dot product between the tensors
y1 = tf.sigmoid((tf.matmul(x, W1) + b1), name='activationLayer1')

# network parameters(weights and biases) are set and initialized (Layer2)
W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1),
                 name='weights2')
b2 = tf.Variable(tf.random_normal([n_neurons_in_h2], mean=0, stddev=1), name='biases2')
# activation function(sigmoid)
y2 = tf.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

# output layer weights and biases
Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, labels_dim], mean=0, stddev=1 ),
                 name='weightsOut')
bo = tf.Variable(tf.random_normal([labels_dim], mean=0, stddev=1), name='biasesOut')

# the sigmoid (binary softmax) activation is absorbed into TF's sigmoid_cross_entropy_with_logits loss
logits = (tf.matmul(y2, Wo) + bo)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)

# tap a separate output that applies softmax activation to the output layer
# for training accuracy readout
a = tf.nn.sigmoid(logits, name='activationOutputLayer')

# optimizer used to compute gradient of loss and apply the parameter updates.
# the train_step object returned is ran by a TF Session to train the net

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# prediction accuracy
# compare predicted value from network with the expected value/target

correct_prediction = tf.equal(tf.round(a), y)
# accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

#############################################
# ***NOTE global_variables_initializer() must be called before creating a tf.Session()!***
init_op = tf.global_variables_initializer()

# create a session for training and feedforward (prediction). Sessions are TF's way to run
# feed data to placeholders and variables, obtain outputs and update neural net parameters
with tf.Session() as sess:
    # ***initialization of all variables... NOTE this must be done before running any further sessions!***
    sess.run(init_op)

    # training loop over the number of epochs
    batch_size = 50
    batches = int(len(X_) / batch_size)

    for epoch in range(training_epochs):
        losses = 0
        accs = 0
        for j in range(batches):
            idx = np.random.randint(X_.shape[0], size=batch_size)
            X_b = X_[idx]
            Y_b = Y_[idx]

            # train the network, note the dictionary of inputs and labels
            sess.run(train_step, feed_dict={x: X_b, y: Y_b})
            # feedforwad the same data and labels, but grab the accuracy and loss as outputs
            acc, l, soft_max_a = sess.run([accuracy, loss, a], feed_dict={x: X_b, y: Y_b})

            losses = losses + np.sum(l)
            accs = accs + np.sum(acc)
        print("Epoch %.8d " % epoch, "avg train loss over", batches, " batches ", "%.4f" % (losses/batches),
              "avg train acc ", "%.4f" % (accs/batches))

        # test on the holdout set
        acc, l, soft_max_a = sess.run([accuracy, loss, a], feed_dict={x: X_t, y: Y_t})
        print("Epoch %.8d " % epoch, "test loss %.4f" % np.sum(l),
              "test acc %.4f" % acc)

print(soft_max_a)
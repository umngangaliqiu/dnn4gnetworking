import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
import scipy.io as sio
import scipy
import scipy.stats
from networkmpc import mpc


nm = 41
no_pv = 5
pf = 0.9
alpha = 0.1
beta = 0.1

no_trajectories = 1
no_iteration = 10
bus, branch = mpc(pf, beta)
from_to = branch[:, 0:2]
#pv_bus = np.asarray([bus[1, 11], bus[14, 11], bus[15, 11], bus[17, 11], bus[18, 11]])
pv_set = np.asarray([1, 14, 15, 17, 18])
qg_min_temp, qg_max_temp = np.float32(bus[pv_set, 12]), np.float32(bus[pv_set, 11])
qg_min = qg_min_temp.T
qg_max = qg_max_temp.T

r_vector = np.zeros((nm, 1))
x_vector = np.zeros((nm, 1))
A_tilde = np.zeros((nm, nm+1))

for i in range(nm):
    A_tilde[i, i+1] = -1
    for k in range(nm):
        if branch[k, 1] == i + 1:
            A_tilde[i, int(from_to[k, 0])] = 1
            r_vector[i] = branch[k, 2]
            x_vector[i] = branch[k, 3]

a0 = A_tilde[:, 0]
A = A_tilde[:, 1:]
A_inv = np.linalg.inv(A)

R_matrix = np.diagflat(r_vector)
X_matrix = np.diagflat(x_vector)
v0 = np.ones(1)

# load data
n_load = sio.loadmat("bus_47_load_data.mat")
n_solar = sio.loadmat("bus_47_solar_data.mat")
load_data = n_load['bus47loaddata']
solar_data = n_solar['bus47solardata']

pc, pg, qc = preprocess_data(load_data, solar_data, bus, alpha)  # pc pg qc all 41*1000
p = pg - pc  # 41*1000
data_set_temp = np.vstack((p, qc))
data_set = data_set_temp.T  # 1000*82


n_neurons_in_h1 = 100
n_neurons_in_h2 = 100
learning_rate = 0.003

def nn_get_loss(data_input):
    train_x = np.asarray(data_input)
    train_x = np.reshape(train_x, (1, 1, 2*nm))
    x = tf.placeholder(tf.float32, [None, 1, 2*nm])
    W1 = tf.Variable(tf.truncated_normal([2*nm, n_neurons_in_h1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([n_neurons_in_h1]))
    W2 = tf.Variable(tf.truncated_normal([n_neurons_in_h1, n_neurons_in_h2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([n_neurons_in_h2]))
    W3 = tf.Variable(tf.truncated_normal([n_neurons_in_h2, 2*no_pv], stddev=0.1))
    b3 = tf.Variable(tf.zeros([2*no_pv]))

    y1 = (tf.matmul(tf.reshape(x, [-1, 82]), W1) + b1)
    y2 = (tf.matmul(y1, W2) + b2)
    y3 = (tf.matmul(y2, W3) + b3)

    q_sample = np.zeros((1, nm))
    mean = y3[:, :no_pv]
    std = y3[:, no_pv:]

    lower_bound = (tf.stack([qg_min for _ in range(no_pv)]) - mean) / std
    upper_bound = (tf.stack([qg_max for _ in range(no_pv)]) - mean) / std

    qg_draw = scipy.stats.truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std)
    for j in range(no_pv):
        q_sample[pv_set[j]] = qg_draw[j]

    reward_cvx = cvx_dc(data_input, q_sample, r_vector, R_matrix, X_matrix, A, A_inv, a0, v0, bus, nm)

 #   trunc_pdf = truncnorm.pdf(qg_draw, qg_min, qg_max, loc=mean_var[0:no_pv, :], scale=y3[no_pv:, :])

    mean_pdf = tf.nn.sigmoid(mean) * (qg_max - qg_min) + qg_min
    std_pdf = tf.nn.sigmoid(std) * np.sqrt(qg_max - qg_min) + 0.05  # TODO: add a little epsilon?

    dist = tf.distributions.Normal(mean_pdf, std_pdf)
    log_probs = dist.log_prob(qg_draw) - tf.log(dist.cdf(qg_max) - dist.cdf(qg_min))
    log_probs = tf.reduce_sum(log_probs, axis=1)

    loss_func = tf.reduce_sum(reward_cvx*tf.log(log_probs))
 #   loss_func = tf.reduce_sum(tf.log(log_probs))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss_func)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(no_iteration):
        train_data = {x: train_x}
        # _,c1=sess.run([train_step, cross_entropy], feed_dict=train_data)
        sess.run(train_step, feed_dict=train_data)
        qg_optimal_temp = sess.run([loss_func], feed_dict=train_data)


        if i == no_iteration-1:
            qg_optimal.append(qg_optimal_temp)
            print(qg_optimal_temp)

    return qg_optimal

qg_draw = np.zeros(no_pv)
reward_cvx = 0
qg_optimal = []
for k in range(1000):
    qg_optimal = nn_get_loss(data_set[k])


plt.plot(qg_optimal, 'ro')
plt.ylabel('loss')
plt.xlabel('number of example')
plt.show()
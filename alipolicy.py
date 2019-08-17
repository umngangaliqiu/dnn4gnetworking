from utils import *
from keras import backend as ke
from keras.layers import Input, Dense
from keras import losses
from keras.models import Model
import numpy as np

from networkmpc import mpc
import scipy.io as sio
import tensorflow as tf

from scipy.stats import truncnorm


from scipy.stats import multivariate_normal

from keras import optimizers
from scipy.stats import truncnorm
from matplotlib import pyplot as plt

nm = 41
no_pv = 5


total_iteration = 100

# load mpc
pf = 0.8
alpha = 0.8
beta = 0.2
bus, branch = mpc(pf, beta)
from_to = branch[:, 0:2]
pv_bus = np.array([bus[1, 11], bus[14, 11], bus[15, 11], bus[17, 11], bus[18, 11]])
pv_set = np.array([1, 14, 15, 17, 18])
qg_min, qg_max = np.float32(bus[pv_set, 12]), np.float32(bus[pv_set, 11])


r = np.zeros((nm, 1))
x = np.zeros((nm, 1))
A_tilde = np.zeros((nm, nm+1))

for i in range(nm):
    A_tilde[i, i+1] = -1
    for k in range(nm):
        if branch[k, 1] == i + 1:
            A_tilde[i, int(from_to[k, 0])] = 1
            r[i] = branch[k, 2]
            x[i] = branch[k, 3]

a0 = A_tilde[:, 0]
A = A_tilde[:, 1:]
A_inv = np.linalg.inv(A)
R = np.diagflat(r)
X = np.diagflat(x)
v0 = np.ones(1)

# load data
n_load = sio.loadmat("bus_47_load_data.mat")
n_solar = sio.loadmat("bus_47_solar_data.mat")
load_data = n_load['bus47loaddata']
solar_data = n_solar['bus47solardata']

pc, pg, qc = preprocess_data(load_data, solar_data, bus, alpha)
p = pg - pc
data_set_temp = np.vstack((p, qc))
data_set = data_set_temp.T
print(np.shape(data_set))

lower = np.zeros((no_pv, 1))
upper = pv_bus


#############################################################################
def draw_sample(mean_var):
    sh = np.shape(mean_var)[0]
    qg_draw = np.zeros((sh, no_pv))
    for i_temp in range(sh):
        for j in range(no_pv):
            a, b = (qg_min[j] - mean_var[i_temp, j])/np.abs(mean_var[i_temp, no_pv+j]), \
                   (qg_max[j] - mean_var[i_temp, j])/np.abs(mean_var[i_temp, no_pv+j])
            qg_draw[i_temp, j] = truncnorm.rvs(a, b, loc=mean_var[i_temp, j],
                                               scale=np.abs(mean_var[i_temp, no_pv+j]), size=1)
    return np.float32(qg_draw)

    
#############################################################################


no_trajectories = 3
H1, H3 = 50, 2*no_pv
epoch = 1

x_dim = np.shape(data_set)[1]
inputs = Input(shape=(x_dim,))


# Define custom loss
def customized_loss(tr_qg):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss_dual(y_true, y_pred):
        log_pi = np.zeros(no_trajectories)

        for i_col in range(no_trajectories):
            log_pi[i_col] = truncnorm.pdf(tr_qg, qg_min, qg_max, loc=y_pred[i_col, 0:no_pv], scale=y_pred[i_col, no_pv:])
        return 0 * ke.mean(y_true-y_pred) + np.sum(y_true * truncnorm.pdf(tr_qg, qg_min,
                                                                                qg_max, loc=y_pred[i_col, 0:no_pv],
                                             scale=y_pred[:, no_pv:]))

    return loss_dual


obj_local = np.zeros((no_trajectories, 1))
qg_local = np.zeros((no_trajectories, no_pv))

x1 = Dense(H1, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(inputs)
# x2 = Dense(H2, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(x1)
predictions = Dense(H3, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros')(x1)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='SGD', loss=customized_loss(qg_local))

features = np.random.normal(loc=0, scale=1.0, size=(no_trajectories, x_dim))
labels = np.zeros((no_trajectories, 2*no_pv))
model.fit(features, labels, epochs=epoch)


for iterations in range(total_iteration):

    x_local = data_set[no_trajectories*iterations:no_trajectories*(iterations+1), :]
    p_local = x_local[:, 0:nm]
    qc_local = x_local[:, nm:]

    predicted_output = model.predict(x=x_local)
    predicted_mu = predicted_output[:, 0:no_pv]
    predicted_sigma = predicted_output[:, no_pv:]

    for i in range(no_trajectories):

        qg_local = draw_sample(predicted_output)
        q_local = -qc_local

        q_local[i, pv_set] = qg_local[i, :] - qc_local[i, pv_set]
        obj_local[i] = cvx_fun(p_local[i, :], q_local[i, :], r, R, X, A, A_inv, a0, v0, bus, nm)
        print(obj_local[i])

    model.compile(optimizer="Adam", loss=customized_loss(qg_local))
    weights = model.get_weights()

    model.set_weights(weights)
    # labels = np.zeros((no_trajectories, 2*no_pv))
    y_true = obj_local
    model.fit(x_local, y_true, epochs=epoch)


# # print("Train is running ...")
# # model.fit(features, labels, batch_size=1, epochs=epoch, shuffle=False, class_weight
#
#
# # model.add_loss()
#
# feature_test = np.random.normal(loc=0, scale=10.0, size=(no_test_samples, x_dim))
# label_test = np.sign(np.tensordot(feature_test, true_theta, axes=([1], [1])))
#
# # print("Tests is running ...")
# # y_pred = model(feature_test)
# # print(np.tensordot(y_pred, label_test, axes=([0], [0])))
# prediction_error = np.sum(np.abs(np.sign(model.predict(x=feature_test))-label_test))
# # print(model.predict(x=feature_test)[0])
# # print("\n")
#
# # print([np.sign(model.predict(x=feature_test)), label_test])
#
# print("true labels \n")
# print(label_test)
# print("predicted labels \n")
# print(model.predict(x=feature_test))
# # print("prediction_error is %f" %prediction_error)

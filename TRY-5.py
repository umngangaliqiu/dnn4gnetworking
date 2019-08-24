from utils import *
import numpy as np
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import optimizers
from matplotlib import pyplot as plt
import scipy.io as sio
from networkmpc import mpc
from scipy.stats import truncnorm
import tensorflow as tf
import copy

SEED = 12
np.random.seed(SEED)

nm = 41
no_pv = 5
pf = 0.8
alpha = 0.05
beta = 0.05
no_trajectories = 1
learning_rate = 0.001

train_size = 8000

bus, branch = mpc(pf, beta)
from_to = branch[:, 0:2]
# pv_bus = np.array([bus[0, 11], bus[13, 11], bus[14, 11], bus[16, 11], bus[17, 11]])
pv_set = np.array([1, 14, 15, 17, 18])
cap_set = np.array([2, 31, 41]) # this numbering referred to the bus mps wich has 42 rows
qg_min, qg_max = np.float32(bus[pv_set, 12]), np.float32(bus[pv_set, 11])
v_max = bus[1:, 3]
v_min = bus[1:, 4]
r_vector = np.zeros(nm)
x_vector = np.zeros(nm)

A_tilde = np.zeros((nm, nm+1))

for i in range(nm):
    A_tilde[i, i+1] = -1
    for k in range(nm):
        if branch[k, 1] == i + 2:
            A_tilde[i, int(from_to[k, 0])-1] = 1
            r_vector[i] = branch[k, 2]
            x_vector[i] = branch[k, 3]

a0 = A_tilde[:, 0]
a_matrix = A_tilde[:, 1:]
a_inv = np.linalg.inv(a_matrix)
r_matrix = np.diagflat(r_vector)
x_matrix = np.diagflat(x_vector)
v0 = np.ones(1)

# cvx_ac(p, q, r, x, nm, v_max, v_min)

# load data
n_load = sio.loadmat("bus_47_load_data.mat")
n_solar = sio.loadmat("bus_47_solar_data.mat")
load_data = n_load['bus47loaddata']
solar_data = n_solar['bus47solardata']

permutation_indices = np.random.permutation(10000)

load_data_train = load_data[permutation_indices[:train_size], :]
solar_data_train = solar_data[permutation_indices[:train_size], :]


load_data_test = load_data[permutation_indices[train_size:], :]
solar_data_test = solar_data[permutation_indices[train_size:], :]


pc_train, pg_train, qc_train = pre_process_data(load_data_train, solar_data_train, bus, alpha)  # pc_train pg_train qc_train all 41*10000
p_train = pg_train - pc_train  # 41*10000
data_set_temp_train = np.vstack((p_train, qc_train))
data_set_train = data_set_temp_train.T  # 10000*82
# aq, av, bv, an, bn, cn, dn = pre_process_cvx_ac_matrix(p_train, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, nm)
pc_test, pg_test, qc_test = pre_process_data(load_data_test, solar_data_test, bus, alpha)  # pc_train pg_train qc_train all 41*10000
p_test = pg_test - pc_test  # 41*10000
data_set_temp_test = np.vstack((p_test, qc_test))
data_set_test = data_set_temp_test.T  # 10000*82

class Agent(object):

    def __init__(self, input_dim, output_dim, hidden_dims):
        """Gym Playing Agent
        Args:
            input_dim (int): the dimension of state.
                Same as `env.observation_space.shape[0]`
            output_dim (int): the number of discrete actions
                Same as `env.action_space.n`
            hidden_dims (list): hidden dimensions
        Methods:
            private:
                __build_train_fn -> None
                    It creates a train function
                    It's similar to defining `train_op` in Tensorflow
                __build_network -> None
                    It create a base model
                    Its output is each action probability
            public:
                get_action(state) -> action
                fit(state, action, reward) -> None
        """

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.__build_network()
        self.__build_train_fn()

    def __build_network(self):
        """Create a base network"""
        self.X = layers.Input(shape=(self.input_dim,))
        net = self.X

        for h_dim in self.hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(self.output_dim)(net)
        net = layers.Activation("softmax")(net)

        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim/2), name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,), name="discount_reward")

        mu = action_prob_placeholder[:, :no_pv]
        var = action_prob_placeholder[:, no_pv:]

        mean_pdf = tf.nn.sigmoid(mu) * (qg_max - qg_min) + qg_min
        std_pdf = tf.nn.sigmoid(var) * np.sqrt(qg_max - qg_min) + 0.05  # TODO: add a little epsilon?

        dist = tf.distributions.Normal(mean_pdf, std_pdf)
        log_probs = dist.log_prob(action_onehot_placeholder) - tf.log(dist.cdf(qg_max) - dist.cdf(qg_min))
        loss = tf.reduce_sum(discount_reward_placeholder * tf.log(log_probs))

        # action_prob = action_prob_placeholder
        # log_action_prob = K.log(action_prob)
        # loss = - log_action_prob * discount_reward_placeholder
        # loss = K.mean(loss)

        adam = optimizers.adam(lr=learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)
        return loss

    def get_action(self, state):

        shape = state.shape
        if len(shape) == 1:
            assert shape == (self.input_dim,), "{} != {}".format(shape, self.input_dim)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == self.input_dim, "{} != {}".format(shape, self.input_dim)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        qg_draw = np.zeros(no_pv)
        mean_var = np.squeeze(self.model.predict(state))
        for j in range(no_pv):
            a, b = (qg_min[j] - mean_var[j]) / np.abs(mean_var[no_pv + j]), \
                   (qg_max[j] - mean_var[j]) / np.abs(mean_var[no_pv + j])
            qg_draw[j] = truncnorm.rvs(a, b, loc=mean_var[j],
                                       scale=np.abs(mean_var[no_pv + j]), size=1)

        # action_prob = np.squeeze(self.model.predict(state))
        return qg_draw  # np.random.choice(np.arange(self.output_dim), p=action_prob)

    # def fit(self, ss, aa, rr):
    #     """Train a network
    #     Args:
    #         ss (2-D Array): `state` array of shape (n_samples, state_dimension)
    #         aa (1-D Array): `action` array of shape (n_samples,)
    #             It's simply a list of int that stores which actions the agent chose
    #         rr (1-D Array): `reward` array of shape (n_samples,)
    #             A reward is given after each action.
    #             :param aa:
    #     """
    #     action_onehot = aa
    #     discount_reward = rr
    #
    #     # assert ss.shape[1] == self.input_dim, "{} != {}".format(ss.shape[1], self.input_dim)
    #     # assert action_onehot.shape[0] == ss.shape[0], "{} != {}".format(action_onehot.shape[0], ss.shape[0])
    #     # assert action_onehot.shape[1] == self.output_dim, "{} != {}".format(action_onehot.shape[1], self.output_dim)
    #     # assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))
    #
    #     self.train_fn([ss, aa, discount_reward])


def run_episode(agent, episode, data_sample, is_train):

    set_state = []
    set_action = []
    set_reward = []
    # r_init = np.random.randint(0, train_size)
    s = copy.deepcopy(data_sample)
    print(s)
    total_reward = 0

    # done = False

    # while not done:

    a = agent.get_action(s)  # No_pv
    # s2 = data_set[episode+1, :]

    p_sample = s[:nm]
    q_sample = -s[nm:]
    q_sample[pv_set - 1] = a + q_sample[pv_set - 1]
    q_sample[cap_set - 1] = bus[cap_set - 1, 11]

    rr = np.squeeze(cvx_ac(p_sample, q_sample, r_vector, x_vector, a_inv, a_matrix, r_matrix, x_matrix, v_min, v_max, nm, a0, v0, branch, bus))

    total_reward += rr

    set_state.append(s)
    set_action.append(a)
    set_reward.append(rr)

    # s = s2

    # done = True

    if is_train:
        set_state = np.array(set_state)
        set_action = np.array(set_action)
        set_reward = np.array(set_reward)
        agent.train_fn([set_state, set_action, set_reward])

    return total_reward


def main():
    try:
        episode_train_no = np.minimum(train_size, 100)
        episode_test_no = np.minimum(train_size, 100)

        accu_reward_train = np.zeros(episode_train_no)
        average_cost_train = np.zeros(episode_train_no)

        accu_reward_test = np.zeros(episode_test_no)
        average_cost_test = np.zeros(episode_test_no)

        accu_reward_cvx_qg = np.zeros(episode_test_no)
        average_cost_cvx_qg = np.zeros(episode_test_no)

        input_dim = nm * 2
        output_dim = no_pv * 2
        hidden_dim = np.array([nm*2, nm*2, nm*2, nm*2])
        agent = Agent(input_dim, output_dim, hidden_dim)

        history_q = np.ndarray((2 * episode_test_no, nm))

        for episode_train in range(episode_train_no):
            data_train_sample = np.squeeze(data_set_train[episode_train, :])
            reward_train = run_episode(agent, episode_train, data_train_sample, is_train=True)
            accu_reward_train[episode_train] = reward_train
            average_cost_train[episode_train] = np.sum(accu_reward_train)/(episode_train+1)
            print(episode_train, reward_train)

        for episode_test in range(episode_test_no):
            data_test_sample = np.squeeze(data_set_test[episode_test, :])
            reward_test = run_episode(agent, episode_test, data_test_sample, is_train=False)
            accu_reward_test[episode_test] = reward_test
            average_cost_test[episode_test] = np.sum(accu_reward_test)/(episode_test+1)

            zz = p_test.T[episode_test, :]
            zz2 = qc_test.T[episode_test, :]
            accu_reward_cvx_qg[episode_test] = cvx_ac_qg(p_test.T[episode_test, :], qc_test.T[episode_test, :], r_vector, x_vector, a_inv, a_matrix, r_matrix, x_matrix, v_min, v_max, nm, a0, v0, branch, bus)
            average_cost_cvx_qg[episode_test] = np.sum(accu_reward_cvx_qg)/(episode_test+1)
            history_q[2 * episode_test, :] = qc_test.T[episode_test, :]
            history_q[2 * episode_test + 1, :] = data_set_test[episode_test, nm:]
            print(episode_test, reward_test)

    finally:
        sio.savemat("data/cost_train.mat", {"foo": average_cost_train})
        sio.savemat("data/cost_test.mat", {"foo": average_cost_test})
        sio.savemat("data/cost_cvx_qg.mat", {"foo": average_cost_cvx_qg})

        plt.plot(average_cost_train, 'r-')
        plt.plot(average_cost_test, 'b-')
        plt.plot(average_cost_cvx_qg, 'k-')
        plt.legend(['Train', 'Test', 'cvx_qg'], loc='upper right', prop={'size': 10})
        plt.show()


if __name__ == '__main__':
    main()

from utils import *
import numpy as np
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from matplotlib import pyplot as plt
import scipy.io as sio
from utils import preprocess_data
from networkmpc import mpc
from scipy.stats import truncnorm

nm = 41
no_pv = 5
pf = 0.9
alpha = 0.1
beta = 0.1

no_trajectories = 1
learning_rate = 0.001

bus, branch = mpc(pf, beta)
from_to = branch[:, 0:2]
pv_bus = np.array([bus[1, 11], bus[14, 11], bus[15, 11], bus[17, 11], bus[18, 11]])
pv_set = np.array([1, 14, 15, 17, 18])
qg_min, qg_max = np.float32(bus[pv_set, 12]), np.float32(bus[pv_set, 11])
v_max = bus[1:, 3]
v_min = bus[1:, 4]
r = np.zeros(nm)
x = np.zeros(nm)

A_tilde = np.zeros((nm, nm+1))

for i in range(nm):
    A_tilde[i, i+1] = -1
    for k in range(nm):
        if branch[k, 1] == i + 2:
            A_tilde[i, int(from_to[k, 0])-1] = 1
            r[i] = branch[k, 2]
            x[i] = branch[k, 3]
print(r)
print(x)
a0 = A_tilde[:, 0]
A = A_tilde[:, 1:]
print(A)
A_inv = np.linalg.inv(A)
print(A)
R = np.diagflat(r)
X = np.diagflat(x)
v0 = np.ones(1)

# cvx_ac(p, q, r, x, nm, v_max, v_min)

# load data
n_load = sio.loadmat("bus_47_load_data.mat")
n_solar = sio.loadmat("bus_47_solar_data.mat")
load_data = n_load['bus47loaddata']
solar_data = n_solar['bus47solardata']

pc, pg, qc = preprocess_data(load_data, solar_data, bus, alpha)  # pc pg qc all 41*10000
p = pg - pc  # 41*10000
data_set_temp = np.vstack((p, qc))
data_set = data_set_temp.T  # 10000*82
h_layer1 = 16
h_layer2 = 16

class Agent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[h_layer1, h_layer2]):
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
        qg_prob_placeholder = K.placeholder(shape=(None, self.output_dim/2), name="qg_values")
        discount_reward_placeholder = K.placeholder(shape=(None,), name="discount_reward")

        mu = action_prob_placeholder[:, 0:no_pv]
        var = action_prob_placeholder[:, no_pv:]

        mean_pdf = tf.nn.sigmoid(mu) * (qg_max - qg_min) + qg_min
        std_pdf = tf.nn.sigmoid(var) * np.sqrt(qg_max - qg_min) + 0.05  # TODO: add a little epsilon?

        dist = tf.distributions.Normal(mean_pdf, std_pdf)
        log_probs = dist.log_prob(action_onehot_placeholder) - tf.log(dist.cdf(qg_max) - dist.cdf(qg_min))
        loss = tf.reduce_sum(discount_reward_placeholder * tf.log(log_probs))

        # action_prob = action_prob_placeholder
        #
        # log_action_prob = K.log(action_prob)
        #
        # loss = - log_action_prob * discount_reward_placeholder
        # loss = K.mean(loss)

        adam = optimizers.adam(lr=learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

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

    def fit(self, ss, aa, rr):
        """Train a network
        Args:
            ss (2-D Array): `state` array of shape (n_samples, state_dimension)
            aa (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            rr (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
                :param aa:
        """
        action_onehot = aa
        discount_reward = rr

        # assert ss.shape[1] == self.input_dim, "{} != {}".format(ss.shape[1], self.input_dim)
        # assert action_onehot.shape[0] == ss.shape[0], "{} != {}".format(action_onehot.shape[0], ss.shape[0])
        # assert action_onehot.shape[1] == self.output_dim, "{} != {}".format(action_onehot.shape[1], self.output_dim)
        # assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))

        self.train_fn([ss, action_onehot, discount_reward])


def run_episode(agent):

    done = False

    set_state = []
    set_action = []
    set_reward = []

    r_init = np.random.randint(1, 10000)
    s = data_set[r_init, :]
    total_reward = 0

    while not done:

        a = agent.get_action(s)
        s2 = data_set[r_init+1, :]

        p_sample = s[0:nm]
        q_sample = -s[nm:]
        q_sample[pv_set] = a + q_sample[pv_set]

        #rr = np.squeeze(cvx_dc(p_sample, q_sample, r, R, X, A, A_inv, a0, v0, bus, nm, v_max, v_min))
        #rr = np.squeeze(cvx_ac(p_sample, q_sample, r, x, nm, branch, v_max, v_min))

        rr = np.squeeze(cvx_ac_matrix(p_sample, q_sample, r, R, X, A, A_inv, a0, v0, bus, nm, v_max, v_min))

        total_reward += rr

        done = True

        set_state.append(s)
        set_action.append(a)
        set_reward.append(rr)

        s = s2

        if done:

            set_state = np.array(set_state)
            set_action = np.array(set_action)
            set_reward = np.array(set_reward)

            agent.fit(set_state, set_action, set_reward)

    return total_reward


def main():
    try:
        episode_no = 10000
        accu_reward = np.zeros((episode_no, 1))
        input_dim = nm * 2
        output_dim = no_pv * 2
        agent = Agent(input_dim, output_dim, [16, 16])

        for episode in range(episode_no):

            # x_local = data_set[no_trajectories * episode:no_trajectories * (episode + 1), :].T
            # z = np.squeeze(x_local)
            reward = run_episode(agent)
            accu_reward[episode] = reward
            print(episode, reward)

    finally:
        plt.plot(accu_reward)
        plt.show()


if __name__ == '__main__':
    main()

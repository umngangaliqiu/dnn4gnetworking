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


nm = 41
no_pv = 5
total_iteration = 100
# load mpc
pf = 0.8
alpha = 0.8
beta = 0.2

no_trajectories = 3

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


class Agent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32]):
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
        self.output_dim = output_dim

        self.__build_network(input_dim, output_dim, hidden_dims)
        self.__build_train_fn()

    def __build_network(self, input_dim, output_dim, hidden_dims=[32, 32]):
        """Create a base network"""
        self.X = layers.Input(shape=(input_dim,))
        net = self.X

        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
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

        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def get_action(self, state):

        action_prob = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(self.output_dim), p=action_prob)


    def fit(self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
        discount_reward = compute_discounted_R(R)

        assert S.shape[1] == self.input_dim, "{} != {}".format(S.shape[1], self.input_dim)
        assert action_onehot.shape[0] == S.shape[0], "{} != {}".format(action_onehot.shape[0], S.shape[0])
        assert action_onehot.shape[1] == self.output_dim, "{} != {}".format(action_onehot.shape[1], self.output_dim)
        assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))

        self.train_fn([S, action_onehot, discount_reward])


def compute_discounted_R(R, discount_rate=.99):
    """Returns discounted rewards
    Args:
        R (1-D array): a list of `reward` at each time step
        discount_rate (float): Will discount the future value by this rate
    Returns:
        discounted_r (1-D array): same shape as input `R`
            but the values are discounted
    Examples:
        [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
    """
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean() / discounted_r.std()

    return discounted_r


def run_episode(x_sample, agent):
    """Returns an episode reward
    (1) Play until the game is done
    (2) The agent will choose an action according to the policy
    (3) When it's done, it will train from the game play
    Args:
        env (gym.env): Gym environment
        agent (Agent): Game Playing Agent
    Returns:
        total_reward (int): total reward earned during the whole episode
    """

    p_sample = x_sample[:, 0:nm]
    qc_sample = x_sample[:, nm:]

    for i_sample in range(no_trajectories):

        s = x_sample[i_sample, :]
        a = agent.get_action(s)
        q_sample = -qc_sample

        q_sample[i_sample, pv_set] = a - qc_sample[i_sample, pv_set]
        rr = cvx_fun(p_sample[i_sample, :], q_sample[i_sample, :], r, R, X, A, A_inv, a0, v0, bus, nm)

        agent.fit(s, a, rr)


def main():
    try:
        episode_no = 10000
        accu_reward = np.zeros((episode_no, 1))
        input_dim = nm * 2
        output_dim = no_pv * 2
        agent = Agent(input_dim, output_dim, [16, 16])

        for episode in range(episode_no):

            x_local = data_set[no_trajectories * episode:no_trajectories * (episode + 1), :]
            reward = run_episode(x_local, agent)
            accu_reward[episode] = reward
            print(episode, reward)

    finally:
        plt.plot(accu_reward)
        # plt.show()


if __name__ == '__main__':
    main()

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

from environments.abstract_environment import AbstractEvaluationPolicy, \
    AbstractEnvironment
from utils.np_utils import one_hot_embed


class SimpleLinearEnvironment(AbstractEnvironment):
    def __init__(self, pci_reducer):
        AbstractEnvironment.__init__(self, pci_reducer)
        self.eta, self.y, self.x = None, None, None
        self.num_a = 3
        self.a_vals = [a_ for a_ in range(self.num_a)]
        self.x_dim = 2
        self.u = np.array([[0.5, 0.0], [0.0, 0.9]])
        self.a_mu = 0.0
        self.a_t = 1.0
        self.v = {0: np.array([1.0, 1.0]),
                  1: np.array([1.0, 0]),
                  2: np.array([-1.0, -1.0])}
        self.a_temp = 2.0
        self.alpha = 0.85
        self.beta = np.array([2.0, 0.5])
        self.theta = np.array([0.0, 2.0, -2.0])
        self.discretize_x_thresholds = [-7.0, -4.0, -2.0, -1.0, 0.0,
                                        1.0, 2.0, 4.0, 7.0]

    def reset(self):
        self.sample_init_state()
        a = self.sample_action()
        o_prior = self.discretize_x(self.x)
        self.transition_state(a)
        return o_prior

    def sample_init_state(self):
        self.y = 0
        self.eta = np.random.choice([-1.0, 1.0])
        self.x = np.array([self.eta, self.eta ** 2])

    def sample_new_x(self, a):
        x_noise = np.array([1.0, -3.0]) * self.eta + np.random.randn(2)
        return self.u @ self.x + self.v[a] + 0.2 * x_noise

    def sample_observation(self):
        return self.discretize_x(self.x)

    def discretize_x(self, x):
        code = 0
        digit_len = len(self.discretize_x_thresholds) + 1
        for i in range(self.x_dim):
            digit = sum([x[i] <= t_
                         for t_ in self.discretize_x_thresholds])
            code = digit_len * code + digit
        return code

    @staticmethod
    def euc_dist(a, b):
        return ((a - b) ** 2).sum() ** 0.5

    def sample_action(self):
        a_logits = np.array([-1.0 * self.euc_dist(self.x, self.v[a_])
                             for a_ in range(self.num_a)])
        a_p = softmax(a_logits / self.a_temp)
        return np.random.choice(self.a_vals, p=a_p)

    def transition_state(self, a):
        eps = np.random.randn()
        x_new = self.sample_new_x(a)
        y_new = self.alpha * self.y + self.beta @ x_new + self.theta[a] + eps
        self.x = x_new
        self.y = y_new
        return self.y

    def embed_a(self, a):
        return one_hot_embed(a, self.num_a)

    def embed_o(self, o):
        num_o = int((len(self.discretize_x_thresholds) + 1) ** self.x_dim)
        return one_hot_embed(o, num_o)

    def get_num_a(self):
        return self.num_a

    def zxa_sq_dist(self, zxa_1, zxa_2):
        zxa_1_embed = self.pci_reducer.embed_zxa(zxa_1, self, use_x=False)
        zxa_2_embed = self.pci_reducer.embed_zxa(zxa_2, self, use_x=False)
        return cdist(zxa_1_embed, zxa_2_embed, metric="sqeuclidean")

    def wxa_sq_dist(self, wxa_1, wxa_2):
        wxa_1_embed = self.pci_reducer.embed_wxa(wxa_1, self, use_x=False)
        wxa_2_embed = self.pci_reducer.embed_wxa(wxa_2, self, use_x=False)
        return cdist(wxa_1_embed, wxa_2_embed, metric="sqeuclidean")


class FixedActionVectorPolicy(AbstractEvaluationPolicy):
    def __init__(self, action_vector):
        AbstractEvaluationPolicy.__init__(self, num_a=3)
        self.action_vector = action_vector

    def get_e_t(self, t, o_t, prev_o_t, prev_a_t, prev_r_t):
        a_t = self.action_vector[t]
        return np.array([a_t for _ in range(len(o_t))])

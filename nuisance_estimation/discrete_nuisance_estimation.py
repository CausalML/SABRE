from functools import partial
from collections import defaultdict, Counter
import copy

import numpy as np
import torch
import torch.nn.functional as F

from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.np_utils import is_dicrete_vector_np, one_hot_embed
from utils.torch_utils import torch_to_np, np_to_tensor


def loss_func(m, b, x, freqs, target_mean, min_val, max_val, alpha_vec):
    alpha_1, alpha_2, alpha_3, alpha_4 = alpha_vec
    loss = ((m @ x - b) ** 2).mean()
    if alpha_1 > 0:
        loss = loss + alpha_1 * (freqs * (x ** 2)).sum()
    if alpha_2 > 0:
        loss = loss + alpha_2 * ((freqs * x).sum() - target_mean) ** 2
    if alpha_3 > 0:
        loss = loss + alpha_3 * ((x ** 2).mean())
    if alpha_4 > 0:
        excess = (F.relu(x - max_val) + F.relu(min_val - x))
        loss = loss + alpha_4 * ((freqs * excess) ** 2).sum()
    return loss


def optimize_discrete_obj(x_net_class, x_net_args, x_inputs, m, b,
                          freqs, target_mean, min_val, max_val, alpha_vec):
    try:
        x_net = x_net_class(x_len=x_inputs.shape[-1], a_len=0, num_a=1,
                            **x_net_args)
        optim = torch.optim.LBFGS(params=x_net.parameters())

        def closure():
            optim.zero_grad()
            x = x_net(x_inputs).flatten()
            loss = loss_func(m, b, x, freqs, target_mean, min_val, max_val,
                             alpha_vec)
            loss.backward()
            return loss
        optim.step(closure)
        x = x_net(x_inputs).detach().flatten()
        assert not torch.isnan(x).any()
        return x_net

    except:
        # backup, do SGD
        # print("BACKUP")
        x_net = x_net_class(x_len=x_inputs.shape[-1], a_len=0, num_a=1,
                            **x_net_args)
        optim = torch.optim.Adam(params=x_net.parameters(), lr=5e-2)

        num_iter = 100000
        test_freq = 1000
        max_no_improve = 5
        num_no_improve = 0
        min_loss = float("inf")
        best_x = x_net(x_inputs).detach().flatten()
        for i in range(num_iter):
            optim.zero_grad()
            x = x_net(x_inputs).flatten()
            loss = loss_func(m, b, x, freqs, target_mean, min_val, max_val,
                             alpha_vec)
            loss.backward()
            optim.step()
            if i % test_freq == 0:
                eval_loss = float(loss)
                # print("loss:", eval_loss)
                if torch.isnan(x.data).any():
                    # print("x is nan")
                    break
                elif eval_loss < min_loss + 1e-3:
                    num_no_improve = 0
                    min_loss = eval_loss
                    best_x = x.detach()
                else:
                    num_no_improve += 1
                    if num_no_improve == max_no_improve:
                        # print("no improve")
                        break
        return x_net


def safe_normalize(x, dim=0, eps=1e-2):
    x = np.clip(x, 0, None)
    x_sum = x.sum(axis=dim, keepdims=True)
    other_dim = tuple(i_ for i_ in range(len(x.shape)) if i_ != dim)
    default = x.mean(axis=other_dim, keepdims=True)
    x = (x_sum == 0) * default + x
    x_mean = x / x.sum(axis=dim, keepdims=True)
    unif = np.ones_like(x_mean) / x.shape[dim]
    x_norm = (1 - eps) * x_mean + eps * unif
    return x_norm


class DiscreteNuisanceEstimation(GeneralSequentialNuisanceEstimation):
    def __init__(self, embed_z, embed_w, embed_a, horizon, gamma, num_a,
                 q_net_class, q_net_args, q_alpha_vec,
                 h_net_class, h_net_args, h_alpha_vec):

        q_args = {
            "q_net_class": q_net_class,
            "q_net_args": q_net_args,
            "alpha_vec": q_alpha_vec,
        }
        h_args = {
            "h_net_class": h_net_class,
            "h_net_args": h_net_args,
            "alpha_vec": h_alpha_vec,
        }
        GeneralSequentialNuisanceEstimation.__init__(
            self, embed_z=embed_z, embed_w=embed_w, embed_a=embed_a,
            horizon=horizon, gamma=gamma, num_a=num_a,
            q_class=DiscreteQEstimation, q_args=q_args,
            h_class=DiscreteHEstimation, h_args=h_args)


class DiscreteQEstimation(AbstractQEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a, alpha_vec,
                 q_net_class, q_net_args, eps=1e-2, cuda=False, device=None):
        AbstractQEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)
        self.q_net_class = q_net_class
        self.q_net_args = q_net_args
        self.eps = eps
        self.alpha_vec = alpha_vec
        self.cuda = cuda
        self.device = device

    def fit(self, eta_prev, z_t, w_t, a_t):
        # check that z and w are valid discrete
        assert is_dicrete_vector_np(z_t)
        assert is_dicrete_vector_np(w_t)

        q_net_list = []

        z_vals = np.array(sorted(set(z_t)))
        w_vals = np.array(sorted(set(w_t)))
        z_reverse_index = {z_: i_ for i_, z_ in enumerate(z_vals)}
        w_reverse_index = {w_: i_ for i_, w_ in enumerate(w_vals)}
        z_codes = np.array([z_reverse_index[z_] for z_ in z_t])
        w_codes = np.array([w_reverse_index[w_] for w_ in w_t])
        num_z, num_w = len(z_vals), len(w_vals)

        w_ind = one_hot_embed(w_codes, num_categories=num_w)
        z_ind = one_hot_embed(z_codes, num_categories=num_z)
        iws = eta_prev.flatten()

        w_mean = safe_normalize(w_ind, dim=0, eps=1e-2)
        z_mean = safe_normalize(z_ind, dim=0, eps=1e-2)

        eta_a_w = []
        for a in range(self.num_a):
            eta_a_w.append(((a_t == a) * iws) @ w_mean)
        p_a_given_w = safe_normalize(np.stack(eta_a_w, axis=1), dim=1,
                                     eps=self.eps)

        eta_a_z = []
        for a in range(self.num_a):
            eta_a_z.append(((a_t == a) * iws) @ z_mean)
        p_a_given_z = safe_normalize(np.stack(eta_a_z, axis=1), dim=1,
                                     eps=self.eps)
        p_z = safe_normalize(iws @ z_ind, dim=0, eps=self.eps)

        # compute q vector for each a value
        for a in range(self.num_a):
            # compute P(Z | W, A=a)
            # w_a_mean = w_ind * (a_t == a).reshape(-1, 1)
            # w_a_mean = w_a_mean / w_a_mean.sum(0, keepdims=True)
            w_ind_a = w_ind * (a_t == a).reshape(-1, 1)
            w_a_mean = safe_normalize(w_ind_a, dim=0, eps=self.eps)
            # eta_given_w_a = iws @ w_a_mean
            eta_z_given_w_a_list = []
            for z in range(num_z):
                eta_z_given_w_a_list.append((iws * (z_codes == z)) @ w_a_mean)
            eta_z_given_w_a = np.stack(eta_z_given_w_a_list, axis=1)
            # eta_z_given_w_a = np.clip(eta_z_given_w_a, 0, None)
            # norm = eta_z_given_w_a.sum(1, keepdims=True)
            # p_z_given_w_a = eta_z_given_w_a / norm
            p_z_given_w_a = safe_normalize(eta_z_given_w_a, dim=1, eps=self.eps)

            # solve linear system for q_a
            m = p_z_given_w_a * p_a_given_w[:, a].reshape(-1, 1)
            b = np.ones(num_w)
            z_freqs = p_a_given_z[:, a] * p_z
            target_mean = 1.0

            m_torch, b_torch = self._to_tensor(m), self._to_tensor(b)
            z_freqs_torch = self._to_tensor(z_freqs)
            q_min = 0
            q_max = float((p_a_given_w[:, 0] ** -1).max())

            z_embed = self._to_tensor(self.embed_z(z_vals))
            q_net = optimize_discrete_obj(
                self.q_net_class, self.q_net_args, z_embed, m_torch, b_torch,
                z_freqs_torch, target_mean, q_min, q_max, self.alpha_vec)
            q_net_list.append(q_net)

        # print("q", q_array.data.shape)
        # print(q_array)
        # print("")
        # import time
        # time.sleep(1)

        return partial(self.q_func, q_net_list=q_net_list)

    def q_func(self, z_t, a_t, q_net_list):
        output = np.zeros((len(z_t), 1))
        for a in range(self.num_a):
            q_net = q_net_list[a]
            idx = [i_ for i_, a_ in enumerate(a_t) if a_ == a]
            z_embed = self._to_tensor(self.embed_z(z_t[idx]))
            output[idx] = torch_to_np(q_net(z_embed))
        return output

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class DiscreteHEstimation(AbstractHEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a, alpha_vec,
                 h_net_class, h_net_args, eps=1e-2, cuda=False, device=None):
        AbstractHEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)
        self.eps = eps
        self.h_net_class = h_net_class
        self.h_net_args = h_net_args
        self.alpha_vec = alpha_vec
        self.cuda = cuda
        self.device = device

    def fit(self, eta_prev, nu_t, e_t, y_t, mu_t, h_min, h_max, z_t, w_t, a_t):
        # check that z and w are valid discrete
        assert is_dicrete_vector_np(z_t)
        assert is_dicrete_vector_np(w_t)

        h_net_list = []

        z_vals = np.array(sorted(set(z_t)))
        w_vals = np.array(sorted(set(w_t)))
        z_reverse_index = {z_: i_ for i_, z_ in enumerate(z_vals)}
        w_reverse_index = {w_: i_ for i_, w_ in enumerate(w_vals)}
        z_codes = np.array([z_reverse_index[z_] for z_ in z_t])
        w_codes = np.array([w_reverse_index[w_] for w_ in w_t])
        num_z, num_w = len(z_vals), len(w_vals)

        z_ind = one_hot_embed(z_codes, num_categories=num_z)
        w_ind = one_hot_embed(w_codes, num_categories=num_w)

        iws = eta_prev.flatten()
        w_mean = safe_normalize(w_ind, dim=0, eps=1e-2)
        target = y_t.flatten() * (e_t == a_t)

        eta_a_w = []
        for a in range(self.num_a):
            eta_a_w.append(((a_t == a) * iws) @ w_mean)
        p_a_given_w = safe_normalize(np.stack(eta_a_w, axis=1), dim=1,
                                     eps=self.eps)
        p_w = safe_normalize(iws @ w_ind, dim=0, eps=self.eps)

        # compute h vector for each a value
        for a in range(self.num_a):
            z_ind_a = z_ind * (a_t == a).reshape(-1, 1)
            z_a_mean = safe_normalize(z_ind_a, dim=0, eps=self.eps)

            # compute E[eta_t * target | Z_t, A_t=a]
            target_given_z_a = ((iws * target) @ z_a_mean) / (iws @ z_a_mean)

            # compute P(W | Z, A=a)
            # eta_given_z_a = iws @ z_a_mean
            eta_w_given_z_a_list = []
            for w in range(num_w):
                eta_w_given_z_a_list.append((iws * (w_codes == w)) @ z_a_mean)
            eta_w_given_z_a = np.stack(eta_w_given_z_a_list, axis=1)
            p_w_given_z_a = safe_normalize(eta_w_given_z_a, dim=1, eps=self.eps)
            # eta_w_given_z_a = np.clip(eta_w_given_z_a, 0, None)
            # norm = eta_w_given_z_a.sum(1, keepdims=True)
            # p_w_given_z_a = eta_w_given_z_a / norm
            # p_w_given_z_a = eta_w_given_z_a / eta_given_z_a.reshape(-1, 1)

            # solve linear system for h_a
            m = p_w_given_z_a
            b = target_given_z_a
            w_freqs = p_a_given_w[:, a] * p_w
            target_mean = float((iws * target * (a_t == a)).mean())

            m_torch, b_torch = self._to_tensor(m), self._to_tensor(b)
            w_freqs_torch = self._to_tensor(w_freqs)
            w_embed = self._to_tensor(self.embed_w(w_vals))

            h_net = optimize_discrete_obj(
                self.h_net_class, self.h_net_args, w_embed, m_torch, b_torch,
                w_freqs_torch, target_mean, h_min, h_max, self.alpha_vec)
            h_net_list.append(h_net)

        # print("h", h_array.data.shape)
        # print(h_array)
        # print("")
        # import time
        # time.sleep(1)

        return partial(self.h_func, h_net_list=h_net_list)

    def h_func(self, w_t, a_t, h_net_list):
        output = np.zeros((len(w_t), 1))
        for a in range(self.num_a):
            h_net = h_net_list[a]
            idx = [i_ for i_, a_ in enumerate(a_t) if a_ == a]
            if len(idx) > 0:
                w_embed = self._to_tensor(self.embed_w(w_t[idx]))
                output[idx] = torch_to_np(h_net(w_embed))
        return output

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)

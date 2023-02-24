from functools import partial
from collections import defaultdict
from itertools import product

import numpy as np
import torch

from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.kernels import PairKernel
from utils.torch_utils import torch_to_np, np_to_tensor, torch_to_float


class MMRNuisanceEstimation(GeneralSequentialNuisanceEstimation):
    def __init__(self, embed_z, embed_w, embed_a, horizon, gamma, num_a,
                 q_net_class, q_net_args, f_kernel_class, f_kernel_args,
                 h_net_class, h_net_args, g_kernel_class, g_kernel_args):

        q_class = MMRQEstimation
        h_class = MMRHEstimation
        q_args = {
            "q_net_class": q_net_class,
            "q_net_args": q_net_args,
            "f_kernel_class": f_kernel_class,
            "f_kernel_args": f_kernel_args,
        }
        h_args = {
            "h_net_class": h_net_class,
            "h_net_args": h_net_args,
            "g_kernel_class": g_kernel_class,
            "g_kernel_args": g_kernel_args,
        }
        GeneralSequentialNuisanceEstimation.__init__(
            self, embed_z=embed_z, embed_w=embed_w, embed_a=embed_a,
            horizon=horizon, gamma=gamma, num_a=num_a, q_class=q_class,
            q_args=q_args, h_class=h_class, h_args=h_args)


def hash_np(val):
    if isinstance(val, np.ndarray):
        return tuple(val.flatten())
    else:
        return val


def hash_tuple(val_tuple):
    hash_list = [hash_np(_val) for _val in val_tuple]
    return tuple(hash_list)


def get_unique_tuples(val_lists):
    tuple_dict = defaultdict(int)
    pair_codes = []
    code_vals = []

    for val_tuple in zip(*val_lists):
        tuple_hash = hash_tuple(val_tuple)
        if tuple_hash in tuple_dict:
            pair_code = tuple_dict[tuple_hash]
        else:
            pair_code = len(tuple_dict)
            tuple_dict[tuple_hash] = pair_code
            code_vals.append(val_tuple)
        pair_codes.append(pair_code)

    return pair_codes, code_vals


def get_unique_singles(val_list):
    single_dict = defaultdict(int)
    codes = []
    code_vals = []

    for val in val_list:
        if isinstance(val, np.ndarray):
            val_hash = tuple(val)
        else:
            val_hash = val

        if val_hash in single_dict:
            code = single_dict[val_hash]
        else:
            code = len(single_dict)
            single_dict[val_hash] = code
            code_vals.append(val)
        codes.append(code)

    return codes, code_vals


class MMRQEstimation(AbstractQEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a,
                 q_net_class, q_net_args, f_kernel_class, f_kernel_args,
                 cuda=False, device=None):
        AbstractQEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)

        self.q_net_class = q_net_class
        self.q_net_args = q_net_args
        self.f_kernel = PairKernel(embed_x1=embed_w, embed_x2=embed_a,
                                   base_kernel_class=f_kernel_class,
                                   base_kernel_args=f_kernel_args)

        self.cuda = cuda
        self.device = device

    def q_func(self, z_t, a_t, q_net):
        z_embed = self.embed_z(z_t)
        a_embed = self.embed_a(a_t)
        za_embed = self._to_tensor(np.concatenate([z_embed, a_embed], axis=1))
        return torch_to_np(q_net(za_embed))

    def fit(self, eta_prev, z_t, w_t, a_t):
        n = len(z_t)
        self.f_kernel.train([(w_, a_) for w_ in w_t
                             for a_ in range(self.num_a)])

        # process data to find all unique values for RKHS functions
        za_codes, za_vals = get_unique_tuples([z_t, a_t])
        w_codes, w_vals = get_unique_singles(w_t)
        wa_vals = [(w_val_, a_val_) for w_val_ in w_vals
                   for a_val_ in range(self.num_a)]

        # compute preliminary data structures
        l_f = self.f_kernel(wa_vals, wa_vals)
        j_index = np.zeros((len(za_vals), len(w_vals) * self.num_a))
        for i, za_code in enumerate(za_codes):
            wa_code = a_t[i] + self.num_a * w_codes[i]
            j_index[za_code, wa_code] += eta_prev[i]
        k = self._to_tensor((j_index @ l_f) @ j_index.T)

        # compute second marginalized kernel vector summing over actions
        k_vec = np.zeros(len(za_vals))
        j_index_2 = np.zeros((len(za_vals), len(w_vals) * self.num_a))
        for i, za_code in enumerate(za_codes):
            for a in range(self.num_a):
                wa_code = a + self.num_a * w_codes[i]
                j_index_2[za_code, wa_code] += eta_prev[i]
        j_index_2_vec = j_index_2.sum(0) / n
        k_vec = self._to_tensor((j_index @ l_f) @ j_index_2_vec)
        loss_bias = float(((l_f @ j_index_2.sum(0)) @ j_index_2_vec).sum())

        # compute za embedding
        z_vals_embed = self.embed_z(np.array([z_ for z_, _ in za_vals]))
        a_vals_embed = self.embed_a(np.array([a_ for _, a_ in za_vals]))
        za_vals_embed_torch = self._to_tensor(np.concatenate(
            [z_vals_embed, a_vals_embed], axis=1))

        # set up network and optimizer
        q_net = self.q_net_class(
            **self.q_net_args, x_len=z_vals_embed.shape[-1],
            a_len=a_vals_embed.shape[-1], num_a=self.num_a)
        optimizer = torch.optim.LBFGS(q_net.parameters())

        def closure():
            optimizer.zero_grad()
            q_net_vals = q_net(za_vals_embed_torch) / n
            # print((q_net_vals.T @ k @ q_net_vals).shape)
            loss = (q_net_vals.T @ k @ q_net_vals).sum()
            # print((k_vec.T @ q_net_vals).shape)
            loss = loss - 2.0 * (k_vec @ q_net_vals).sum()

            debug_loss = torch_to_float(loss) + loss_bias
            # print("debug_loss", debug_loss)
            loss.backward()
            return loss

        optimizer.step(closure)

        return partial(self.q_func, q_net=q_net)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class MMRHEstimation(AbstractHEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a,
                 h_net_class, h_net_args, g_kernel_class, g_kernel_args,
                 cuda=False, device=None):
        AbstractHEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)

        self.h_net_class = h_net_class
        self.h_net_args = h_net_args
        self.g_kernel = PairKernel(embed_x1=embed_z, embed_x2=embed_a,
                                   base_kernel_class=g_kernel_class,
                                   base_kernel_args=g_kernel_args)

        self.cuda = cuda
        self.device = device

    def h_func(self, w_t, a_t, h_net):
        w_embed = self.embed_z(w_t)
        a_embed = self.embed_a(a_t)
        wa_embed = self._to_tensor(np.concatenate([w_embed, a_embed], axis=1))
        return torch_to_np(h_net(wa_embed))

    def fit(self, eta_prev, nu_t, e_t, y_t, mu_t, z_t, w_t, a_t):
        n = len(z_t)
        self.g_kernel.train([(z_, a_) for z_, a_ in zip(z_t, a_t)])

        # process data to find all unique values for RKHS functions
        za_codes, za_vals = get_unique_tuples([z_t, a_t])
        wa_codes, wa_vals = get_unique_tuples([w_t, a_t])

        # compute preliminary data structures
        l_g = self.g_kernel(za_vals, za_vals)
        j_index = np.zeros((len(wa_vals), len(za_vals)))
        for i, wa_code in enumerate(wa_codes):
            j_index[wa_code, za_codes[i]] += nu_t[i]
        k = self._to_tensor((j_index @ l_g) @ j_index.T)

        # compute second marginalized kernel vector summing over actions
        j_index_2 = np.zeros((len(wa_vals), len(za_vals)))
        for i, wa_code in enumerate(wa_codes):
            j_index_2[wa_code, za_codes[i]] += mu_t[i]
        j_index_2_vec = j_index_2.sum(0) / n
        k_vec = self._to_tensor((j_index @ l_g) @ j_index_2_vec)
        loss_bias = float(((l_g @ j_index_2.sum(0)) @ j_index_2_vec).sum())

        # compute wa embedding
        w_vals_embed = self.embed_w(np.array([w_ for w_, _ in wa_vals]))
        a_vals_embed = self.embed_a(np.array([a_ for _, a_ in wa_vals]))
        wa_vals_embed_torch = self._to_tensor(np.concatenate(
            [w_vals_embed, a_vals_embed], axis=1))

        # set up network and optimizer
        h_net = self.h_net_class(
            **self.h_net_args, x_len=w_vals_embed.shape[-1],
            a_len=a_vals_embed.shape[-1], num_a=self.num_a)
        optimizer = torch.optim.LBFGS(h_net.parameters())


        def closure():
            optimizer.zero_grad()
            h_net_vals = h_net(wa_vals_embed_torch) / n
            loss = (h_net_vals.T @ k @ h_net_vals).sum()
            # print((h_net_vals.T @ k @ h_net_vals).shape)
            loss = loss - 2.0 * (k_vec.T @ h_net_vals).sum()
            # print((k_vec.T @ h_net_vals).shape)
            debug_loss = torch_to_float(loss) + loss_bias
            # print("debug loss:", debug_loss)
            loss.backward()
            return loss

        optimizer.step(closure)

        return partial(self.h_func, h_net=h_net)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)

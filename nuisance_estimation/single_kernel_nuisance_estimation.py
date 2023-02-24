from functools import partial
from collections import defaultdict, Counter

import numpy as np
import torch

from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.kernels import PairKernel
from utils.torch_utils import torch_to_np, np_to_tensor


class SingleKernelNuisanceEstimation(GeneralSequentialNuisanceEstimation):
    def __init__(self, embed_z, embed_w, embed_a, horizon,
                 gamma, num_a, q_net_class, q_net_args, f_kernel_class,
                 f_kernel_args, h_net_class, h_net_args, g_kernel_class,
                 g_kernel_args, q_alpha, h_alpha, num_rep=2):

        q_class = SingleKernelQEstimation
        h_class = SingleKernelHEstimation
        q_args = {
            "q_net_class": q_net_class,
            "q_net_args": q_net_args,
            "f_kernel_class": f_kernel_class,
            "f_kernel_args": f_kernel_args,
            "alpha": q_alpha,
            "num_rep": num_rep,
        }
        h_args = {
            "h_net_class": h_net_class,
            "h_net_args": h_net_args,
            "g_kernel_class": g_kernel_class,
            "g_kernel_args": g_kernel_args,
            "alpha": h_alpha,
            "num_rep": num_rep
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


class SingleKernelQEstimation(AbstractQEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a,
                 q_net_class, q_net_args, f_kernel_class, f_kernel_args,
                 alpha, num_rep=2, cuda=False, device=None):
        AbstractQEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)

        self.q_net_class = q_net_class
        self.q_net_args = q_net_args
        self.f_kernel = PairKernel(embed_x1=embed_w, embed_x2=embed_a,
                                   base_kernel_class=f_kernel_class,
                                   base_kernel_args=f_kernel_args)

        self.alpha = alpha
        self.num_rep = num_rep
        self.cuda = cuda
        self.device = device

    def fit(self, eta_prev, z_t, w_t, a_t):
        alpha = self.alpha
        while True:
            try:
                # print("TRY Q")
                q_net = self._try_fit_internal(eta_prev, z_t, w_t, a_t, alpha)
                did_succeed = q_net.is_finite()
                # print("DONE TRY Q")
            except Exception as e:
                did_succeed = False

            if did_succeed:
                # print("SUCCEED")
                break
            elif alpha == 0:
                # print("FAIL")
                alpha = 1e-8
            else:
                # print("FAIL")
                alpha *= 10

        return partial(self.q_func, q_net=q_net)

    def q_func(self, z_t, a_t, q_net):
        z_embed = self.embed_z(z_t)
        a_embed = self.embed_a(a_t)
        za_embed = self._to_tensor(np.concatenate([z_embed, a_embed], axis=1))
        return torch_to_np(q_net(za_embed))

    def _try_fit_internal(self, eta_prev, z_t, w_t, a_t, alpha):
        n = len(z_t)
        self.f_kernel.train([(w_, a_) for w_ in w_t
                             for a_ in range(self.num_a)])

        # process data to find all unique values for RKHS functions
        za_codes, za_vals = get_unique_tuples([z_t, a_t])
        w_codes, w_vals = get_unique_singles(w_t)
        wa_vals = [(w_val_, a_val_) for w_val_ in w_vals
                   for a_val_ in range(self.num_a)]

        # compute preliminary data structures
        z_embed = self.embed_z(z_t)
        a_embed = self.embed_a(a_t)
        za_embed = self._to_tensor(np.concatenate([z_embed, a_embed], axis=1))
        l_f = self.f_kernel(wa_vals, wa_vals)
        obs_wa_codes = [a_ + self.num_a * w_code_
                        for a_, w_code_ in zip(a_t, w_codes)]

        q_net = self.q_net_class(**self.q_net_args, x_len=z_embed.shape[-1],
                                 a_len=a_embed.shape[-1], num_a=self.num_a)

        for _ in range(self.num_rep):
            tilde_q_za = torch_to_np(q_net(za_embed))
            # compute weighting matrix for objective
            m = self._to_tensor(self._calc_m_matrix(
                eta_prev, tilde_q_za, l_f, obs_wa_codes, w_codes, alpha))

            # set up tensors for optimization
            eta_prev_torch = self._to_tensor(eta_prev)
            c_2 = np.zeros(len(wa_vals))
            for a_i in range(self.num_a):
                wa_codes_i = [a_i + self.num_a * w_code_ for w_code_ in w_codes]
                c_2 = c_2 + (l_f[wa_codes_i, :].T @ eta_prev.flatten()) / n
            c_2_torch = self._to_tensor(c_2)
            l_f_obs_torch = self._to_tensor(l_f[obs_wa_codes, :])
            z_vals_embed = self.embed_z(np.array([z_ for z_, _ in za_vals]))
            a_vals_embed = self.embed_a(np.array([a_ for _, a_ in za_vals]))
            za_vals_embed_torch = self._to_tensor(np.concatenate(
                [z_vals_embed, a_vals_embed], axis=1))

            optimizer = torch.optim.LBFGS(q_net.parameters(),
                                          line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                rho_sum = torch.zeros(len(wa_vals))
                q_net_vals = q_net(za_vals_embed_torch)
                for code in range(len(za_vals)):
                    idx = [i_ for i_, c_ in enumerate(za_codes) if code == c_]
                    rho_delta = (eta_prev_torch[idx]
                                 * l_f_obs_torch[idx]).sum(0) * q_net_vals[code]
                    rho_sum = rho_sum + rho_delta
                rho = (rho_sum / n) - c_2_torch
                m_rho_x = torch.matmul(m, rho).detach()
                # L2 reg
                # za_freqs = Counter(za_vals)
                # freq_tensor = torch.tensor([za_freqs[za_] for za_ in za_vals])
                # reg = ((q_net_vals * freq_tensor) ** 2).sum() ** 0.5 / n
                # loss = 2.0 * torch.matmul(m_rho_x, rho) + 1e-1 * reg
                loss = 2.0 * torch.matmul(m_rho_x, rho)
                loss.backward()
                return loss

            optimizer.step(closure)

        return q_net

    def _calc_m_matrix(self, eta_prev, tilde_q_za, l_f, obs_wa_codes, w_codes,
                       alpha):
        n = len(eta_prev)

        # calculate C matrix and c_2 array
        c_factor = tilde_q_za * l_f[obs_wa_codes, :]
        for a_i in range(self.num_a):
            wa_codes_i = [a_i + self.num_a * w_code_ for w_code_ in w_codes]
            c_factor = c_factor - l_f[wa_codes_i, :]
        c_factor = eta_prev.reshape(-1, 1) * c_factor
        c_m = (c_factor.T @ c_factor) / n + alpha * l_f
        return np.linalg.inv(c_m)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class SingleKernelHEstimation(AbstractHEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a,
                 h_net_class, h_net_args, g_kernel_class, g_kernel_args,
                 alpha, num_rep=2, cuda=False, device=None):
        AbstractHEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)

        self.h_net_class = h_net_class
        self.h_net_args = h_net_args
        self.g_kernel = PairKernel(embed_x1=embed_z, embed_x2=embed_a,
                                   base_kernel_class=g_kernel_class,
                                   base_kernel_args=g_kernel_args)

        self.alpha = alpha
        self.num_rep = num_rep
        self.cuda = cuda
        self.device = device

    def fit(self, eta_prev, nu_t, e_t, y_t, mu_t, z_t, w_t, a_t):
        alpha = self.alpha
        while True:
            try:
                # print("TRY H")
                h_net = self._try_fit_internal(nu_t, mu_t, z_t, w_t, a_t, alpha)
                did_succeed = h_net.is_finite()
                # print("DONE TRY H")
            except Exception as e:
                did_succeed = False

            if did_succeed:
                # print("SUCCEED")
                break
            elif alpha == 0:
                # print("FAIL")
                alpha = 1e-8
            else:
                # print("FAIL")
                alpha *= 10

        return partial(self.h_func, h_net=h_net)

    def h_func(self, w_t, a_t, h_net):
        w_embed = self.embed_z(w_t)
        a_embed = self.embed_a(a_t)
        wa_embed = self._to_tensor(np.concatenate([w_embed, a_embed], axis=1))
        return torch_to_np(h_net(wa_embed))

    def _try_fit_internal(self, nu_t, mu_t, z_t, w_t, a_t, alpha):
        n = len(z_t)
        self.g_kernel.train([(z_, a_) for z_, a_ in zip(z_t, a_t)])

        # process data to find all unique values for RKHS functions
        za_codes, za_vals = get_unique_tuples([z_t, a_t])
        wa_codes, wa_vals = get_unique_tuples([w_t, a_t])

        # compute preliminary data structures
        w_embed = self.embed_w(w_t)
        a_embed = self.embed_a(a_t)
        wa_embed = self._to_tensor(np.concatenate([w_embed, a_embed], axis=1))
        l_g = self.g_kernel(za_vals, za_vals)
        h_net = self.h_net_class(**self.h_net_args, x_len=w_embed.shape[-1],
                                 a_len=a_embed.shape[-1], num_a=self.num_a)

        for _ in range(self.num_rep):
            tilde_h_wa = torch_to_np(h_net(wa_embed))
            # compute weighting matrix for objective
            m = self._to_tensor(self._calc_m_matrix(
                nu_t, mu_t, tilde_h_wa, l_g, za_codes, alpha))

            # set up tensors for optimization
            nu_t_torch = self._to_tensor(nu_t)
            mu_t_torch = self._to_tensor(mu_t)
            l_g_obs_torch = self._to_tensor(l_g[za_codes, :])
            c_2_torch = (mu_t_torch * l_g_obs_torch).mean(0)
            w_vals_embed = self.embed_w(np.array([w_ for w_, _ in wa_vals]))
            a_vals_embed = self.embed_a(np.array([a_ for _, a_ in wa_vals]))
            wa_vals_embed_torch = self._to_tensor(np.concatenate(
                [w_vals_embed, a_vals_embed], axis=1))

            optimizer = torch.optim.LBFGS(h_net.parameters(),
                                          line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                rho_sum = torch.zeros(len(za_vals))
                h_net_vals = h_net(wa_vals_embed_torch)
                for code in range(len(wa_vals)):
                    idx = [i_ for i_, c_ in enumerate(wa_codes) if code == c_]
                    rho_delta = (nu_t_torch[idx]
                                 * l_g_obs_torch[idx]).sum(0) * h_net_vals[code]
                    rho_sum = rho_sum + rho_delta
                rho = (rho_sum / n) - c_2_torch
                m_rho_x = torch.matmul(m, rho).detach()
                # L2 reg
                # wa_freqs = Counter(wa_vals)
                # freq_tensor = torch.tensor([wa_freqs[wa_] for wa_ in wa_vals])
                # reg = ((h_net_vals * freq_tensor) ** 2).sum() ** 0.5 / n
                # loss = 2.0 * torch.matmul(m_rho_x, rho) + 1e-1 * reg
                loss = 2.0 * torch.matmul(m_rho_x, rho)
                loss.backward()
                return loss

            optimizer.step(closure)

        return h_net

    def _calc_m_matrix(self, nu_t, mu_t, tilde_h_wa, l_g, za_codes, alpha):
        n = len(nu_t)

        # calculate C matrix and c_2 array
        eps = nu_t.reshape(-1, 1) * tilde_h_wa - mu_t.reshape(-1, 1)
        c_factor = eps * l_g[za_codes, :]
        c_m = (c_factor.T @ c_factor) / n + alpha * l_g
        return np.linalg.inv(c_m)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)




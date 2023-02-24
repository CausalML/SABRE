from functools import partial

import numpy as np

from collections import defaultdict

from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.kernels import PairKernel


class DoubleKernelNuisanceEstimation(GeneralSequentialNuisanceEstimation):
    def __init__(self, embed_z, embed_w, embed_a, horizon, gamma, num_a,
                 q_kernel_class, q_kernel_args, f_kernel_class, f_kernel_args,
                 h_kernel_class, h_kernel_args, g_kernel_class, g_kernel_args,
                 q_alpha, q_lmbda, h_alpha, h_lmbda, num_rep=2):

        q_class = DoubleKernelQEstimation
        h_class = DoubleKernelHEstimation
        q_args = {
            "q_kernel_class": q_kernel_class,
            "q_kernel_args": q_kernel_args,
            "f_kernel_class": f_kernel_class,
            "f_kernel_args": f_kernel_args,
            "alpha": q_alpha,
            "lmbda": q_lmbda,
            "num_rep": num_rep
        }
        h_args = {
            "h_kernel_class": h_kernel_class,
            "h_kernel_args": h_kernel_args,
            "g_kernel_class": g_kernel_class,
            "g_kernel_args": g_kernel_args,
            "alpha": h_alpha,
            "lmbda": h_lmbda,
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


class DoubleKernelQEstimation(AbstractQEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a,
                 q_kernel_class, q_kernel_args, f_kernel_class, f_kernel_args,
                 alpha, lmbda, num_rep=2):
        AbstractQEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)

        self.q_kernel = PairKernel(embed_x1=embed_z, embed_x2=embed_a,
                                   base_kernel_class=q_kernel_class,
                                   base_kernel_args=q_kernel_args)
        self.f_kernel = PairKernel(embed_x1=embed_w, embed_x2=embed_a,
                                   base_kernel_class=f_kernel_class,
                                   base_kernel_args=f_kernel_args)

        self.alpha = alpha
        self.lmbda = lmbda
        self.num_rep = num_rep

    def fit(self, eta_prev, z_t, w_t, a_t):
        # tune kernel on (Z,A) and (W,A) pairs
        self.q_kernel.train([(z_, a_) for z_, a_ in zip(z_t, a_t)])
        self.f_kernel.train([(w_, a_) for w_ in w_t
                             for a_ in range(self.num_a)])

        tilde_q = self._dummy_tilde_q
        for _ in range(self.num_rep):
            tilde_q = self._fit_internal(eta_prev, z_t, w_t, a_t, tilde_q)
        return tilde_q

    def _fit_internal(self, eta_prev, z_t, w_t, a_t, tilde_q):
        n = len(z_t)

        # process data to find all unique values for RKHS functions
        za_codes, za_vals = get_unique_tuples([z_t, a_t])
        w_codes, w_vals = get_unique_singles(w_t)
        # wa_codes = [a_ + self.num_a * w_code_ for w_code_ in w_codes
        #             for a_ in range(self.num_a)]
        wa_vals = [(w_val_, a_val_) for w_val_ in w_vals
                   for a_val_ in range(self.num_a)]

        # compute preliminary matrices
        l_q = self.q_kernel(za_vals, za_vals) # + 1e-3 * np.eye(len(za_vals))
        l_f = self.f_kernel(wa_vals, wa_vals) # + 1e-3 * np.eye(len(wa_vals))

        # calculate C matrix and c_2 array
        obs_wa_codes = [a_ + self.num_a * w_code_
                        for a_, w_code_ in zip(a_t, w_codes)]
        c_factor = tilde_q(z_t, a_t) * l_f[obs_wa_codes, :]
        c_2 = np.zeros(len(wa_vals))
        for a_i in range(self.num_a):
            wa_codes_i = [a_i + self.num_a * w_code_ for w_code_ in w_codes]
            c_factor = c_factor - l_f[wa_codes_i, :]
            # c_2 = c_2 + (eta_t.reshape(-1, 1) * l_f[wa_codes_i, :]).mean(0)
            c_2 = c_2 + (l_f[wa_codes_i, :].T @ eta_prev.flatten()) / n
        c_factor = eta_prev.reshape(-1, 1) * c_factor
        c_m = (c_factor.T @ c_factor) / n + self.alpha * l_f

        # calculate Omega matrix
        omega = (eta_prev.reshape(-1, 1, 1)
                 * l_f[obs_wa_codes, :].reshape(n, -1, 1)
                 * l_q[za_codes, :].reshape(n, 1, -1)).mean(0)

        # solve linear system for coefficients beta
        b_m = omega.T @ np.linalg.solve(c_m, omega) + self.lmbda * l_q
        b = omega.T @ np.linalg.solve(c_m, c_2)
        beta = np.linalg.solve(b_m, b)

        return partial(self._kernel_function_q, za_vals=za_vals, beta=beta)

    def _kernel_function_q(self, z_t, a_t, za_vals, beta):
        za_t = [(z_, a_) for z_, a_ in zip(z_t, a_t)]
        return (self.q_kernel(za_t, za_vals) @ beta).reshape(-1, 1)

    def _dummy_tilde_q(self, z_t, a_t):
        n = z_t.shape[0]
        return np.ones((n, 1))


class DoubleKernelHEstimation(AbstractHEstimator):
    def __init__(self, embed_z, embed_w, embed_a, num_a,
                 h_kernel_class, h_kernel_args, g_kernel_class, g_kernel_args,
                 alpha, lmbda, num_rep=2):
        AbstractHEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_a=embed_a, num_a=num_a)

        self.h_kernel = PairKernel(embed_x1=embed_w, embed_x2=embed_a,
                                   base_kernel_class=h_kernel_class,
                                   base_kernel_args=h_kernel_args)
        self.g_kernel = PairKernel(embed_x1=embed_z, embed_x2=embed_a,
                                   base_kernel_class=g_kernel_class,
                                   base_kernel_args=g_kernel_args)

        self.alpha = alpha
        self.lmbda = lmbda
        self.num_rep = num_rep

    def fit(self, eta_prev, nu_t, e_t, y_t, mu_t, h_min, h_max, z_t, w_t, a_t):
        tilde_h = self._dummy_tilde_h
        for _ in range(self.num_rep):
            tilde_h = self._fit_internal(nu_t, mu_t, z_t, w_t, a_t, tilde_h)
        return tilde_h

    def _fit_internal(self, nu_t, mu_t, z_t, w_t, a_t, tilde_h):
        n = len(z_t)

        # tune kernel on (W,A) and (Z,A) pairs
        self.h_kernel.train([(w_, a_) for w_, a_ in zip(w_t, a_t)])
        self.g_kernel.train([(z_, a_) for z_, a_ in zip(z_t, a_t)])

        # process data to find all unique values for RKHS functions
        wa_codes, wa_vals = get_unique_tuples([w_t, a_t])
        za_codes, za_vals = get_unique_tuples([z_t, a_t])

        # compute preliminary matrices
        l_h = self.h_kernel(wa_vals, wa_vals) # + 1e-3 * np.eye(len(wa_vals))
        l_g = self.g_kernel(za_vals, za_vals) # + 1e-3 * np.eye(len(za_vals))

        # calculate C matrix and c_2 array
        # print(nu_t.shape)
        # print(mu_t.shape)
        # print(tilde_h(w_t, a_t).shape)
        eps = nu_t.reshape(-1, 1) * tilde_h(w_t, a_t) - mu_t.reshape(-1, 1)
        c_factor = eps * l_g[za_codes, :]
        c_m = (c_factor.T @ c_factor) / n + self.alpha * l_g
        c_2 = (l_g[za_codes, :].T @ mu_t.flatten()) / n

        # calculate Omega matrix
        omega = (nu_t.reshape(-1, 1, 1)
                 * l_g[za_codes, :].reshape(n, -1, 1)
                 * l_h[wa_codes, :].reshape(n, 1, -1)).mean(0)

        # solve linear system for coefficients beta
        b_m = omega.T @ np.linalg.solve(c_m, omega) + self.lmbda * l_h
        b = omega.T @ np.linalg.solve(c_m, c_2)
        beta = np.linalg.solve(b_m, b)

        return partial(self.kernel_function_h, wa_vals=wa_vals, beta=beta)

    def kernel_function_h(self, w_t, a_t, wa_vals, beta):
        wa_t = [(w_, a_) for w_, a_ in zip(w_t, a_t)]
        return (self.h_kernel(wa_t, wa_vals) @ beta).reshape(-1, 1)

    def _dummy_tilde_h(self, w_t, a_t):
        n = w_t.shape[0]
        return np.ones((n, 1))

from functools import partial
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.np_utils import one_hot_embed
from utils.torch_utils import torch_to_np, np_to_tensor


class DiscreteMMRNuisanceEstimation(GeneralSequentialNuisanceEstimation):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, zxa_sq_dist,
                 wxa_sq_dist, horizon, gamma, num_a, q_net_class, q_net_args,
                 g_kernel_class, g_kernel_args, h_net_class, h_net_args,
                 f_kernel_class, f_kernel_args, q_lmbda, h_lmbda):
        q_class = DiscreteMMRQEstimation
        h_class = DiscreteMMRHEstimation
        q_args = {
            "q_net_class": q_net_class,
            "q_net_args": q_net_args,
            "g_kernel_class": g_kernel_class,
            "g_kernel_args": g_kernel_args,
            "lmbda": q_lmbda,
        }
        h_args = {
            "h_net_class": h_net_class,
            "h_net_args": h_net_args,
            "f_kernel_class": f_kernel_class,
            "f_kernel_args": f_kernel_args,
            "lmbda": h_lmbda,
        }
        GeneralSequentialNuisanceEstimation.__init__(
            self, embed_z=embed_z, embed_w=embed_w, embed_x=embed_x,
            embed_a=embed_a, zxa_sq_dist=zxa_sq_dist, wxa_sq_dist=wxa_sq_dist,
            horizon=horizon, gamma=gamma, num_a=num_a, q_class=q_class,
            q_args=q_args, h_class=h_class, h_args=h_args)


def hash_np(val):
    if isinstance(val, np.ndarray):
        return tuple(val.flatten())
    else:
        return val


def get_unique_vals_codes(data):
    data_hashed = [hash_np(v_) for v_ in data]
    vals = sorted(set(data_hashed))
    reverse_dict = {x_: i_ for i_, x_ in enumerate(vals)}
    codes = [reverse_dict[x_] for x_ in data_hashed]
    return np.array(vals), np.array(codes), reverse_dict


def compute_triplet_embeddings(zxa_vals, wxa_vals, z_t, w_t, x_t, num_a,
                               embed_z, embed_w, embed_x, embed_a):
    z_vals, _, z_map = get_unique_vals_codes(z_t)
    w_vals, _, w_map = get_unique_vals_codes(w_t)
    a_vals, _, a_map = get_unique_vals_codes(np.arange(num_a))
    z_vals_embed = embed_z(z_vals)
    w_vals_embed = embed_w(w_vals)
    a_vals_embed = embed_a(a_vals)

    # pre-compute all required embeddings
    if x_t is None:
        zxa_z = [z_map[z_] for z_, _ in zxa_vals]
        zxa_a = [a_map[a_] for _, a_ in zxa_vals]
        zxa_vals_embed = np.concatenate([z_vals_embed[zxa_z],
                                         a_vals_embed[zxa_a]], axis=1)
        wxa_w = [w_map[w_] for w_, _ in wxa_vals]
        wxa_a = [a_map[a_] for _, a_ in wxa_vals]
        wxa_vals_embed = np.concatenate([w_vals_embed[wxa_w],
                                         a_vals_embed[wxa_a]], axis=1)
    else:
        x_vals, _, x_map = get_unique_vals_codes(x_t)
        x_vals_embed = embed_x(x_vals)
        zxa_z = [z_map[z_] for z_, _, _ in zxa_vals]
        zxa_x = [x_map[x_] for _, x_, _ in zxa_vals]
        zxa_a = [a_map[a_] for _, _, a_ in zxa_vals]
        zxa_vals_embed = np.concatenate([z_vals_embed[zxa_z],
                                         x_vals_embed[zxa_x],
                                         a_vals_embed[zxa_a]], axis=1)
        wxa_w = [w_map[w_] for w_, _, _ in wxa_vals]
        wxa_x = [x_map[x_] for _, x_, _ in wxa_vals]
        wxa_a = [a_map[a_] for _, _, a_ in wxa_vals]
        wxa_vals_embed = np.concatenate([w_vals_embed[wxa_w],
                                         x_vals_embed[wxa_x],
                                         a_vals_embed[wxa_a]], axis=1)
    a_len = a_vals_embed.shape[1]
    return zxa_vals_embed, wxa_vals_embed, a_len


class DiscreteMMRQEstimation(AbstractQEstimator):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, num_a,
                 zxa_sq_dist, wxa_sq_dist, q_net_class, q_net_args,
                 g_kernel_class, g_kernel_args, lmbda, cuda=False, device=None):
        AbstractQEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_x=embed_x, embed_a=embed_a,
                                    num_a=num_a, zxa_sq_dist=zxa_sq_dist,
                                    wxa_sq_dist=wxa_sq_dist)

        self.q_net_class = q_net_class
        self.q_net_args = q_net_args
        self.g_kernel = g_kernel_class(sq_dist_func=wxa_sq_dist,
                                       **g_kernel_args)

        self.lmbda = lmbda
        self.cuda = cuda
        self.device = device

        self.q_net = None

    def fit(self, eta_prev, z_t, w_t, x_t, a_t, e_t):
        # compute codes for each of z, w, x
        n = len(z_t)
        eta_prev = eta_prev / eta_prev.mean()
        if x_t is None:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, a_t))
            wa_data = [(w_, a_) for w_ in w_t for a_ in range(self.num_a)]
            wxa_vals, _, wxa_map = get_unique_vals_codes(wa_data)
            wxa_codes = np.array([wxa_map[tuple(wa_)] for wa_ in zip(w_t, a_t)])
        else:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, x_t, a_t))
            wxa_data = [(w_, x_, a_) for w_, x_ in zip(w_t, x_t)
                        for a_ in range(self.num_a)]
            wxa_vals, _, wxa_map = get_unique_vals_codes(wxa_data)
            wxa_codes = np.array([wxa_map[tuple(wxa_)]
                                  for wxa_ in zip(w_t, x_t, a_t)])

        zxa_vals_embed, wxa_vals_embed, a_len = compute_triplet_embeddings(
            zxa_vals, wxa_vals, z_t, w_t, x_t, self.num_a,
            self.embed_z, self.embed_w, self.embed_x, self.embed_a)
        zx_len = zxa_vals_embed.shape[-1] - a_len
        zxa_vals_torch = self._to_tensor(zxa_vals_embed)

        # train kernel hyper-parameters
        self.g_kernel.train(wxa_vals)

        # set up basic matrices / data structures that are constant for each
        # iteration
        k_reg = 1e-1 * np.eye(len(wxa_vals))
        k = self._to_tensor(self.g_kernel(wxa_vals, wxa_vals) + k_reg)

        zxa_ind = one_hot_embed(zxa_codes, num_categories=len(zxa_vals))
        wxa_ind = one_hot_embed(wxa_codes, num_categories=len(wxa_vals))
        b = self._to_tensor((1/n) * (wxa_ind.T @ zxa_ind))
        c = torch.zeros(len(wxa_vals))
        for a in range(self.num_a):
            if x_t is None:
                wxa_codes_a = np.array([wxa_map[(w_, a)] for w_ in w_t])
            else:
                wxa_codes_a = np.array([wxa_map[(w_, x_, a)]
                                        for w_, x_ in zip(w_t, x_t)])
            wxa_ind_a = one_hot_embed(wxa_codes_a, num_categories=len(wxa_vals))
            c.add_(self._to_tensor(wxa_ind_a.mean(0)))

        freqs = self._to_tensor((eta_prev.reshape(-1, 1) * zxa_ind).mean(0))
        e_a_t = (e_t == a_t).astype(int).reshape(-1, 1)
        reg_freqs = self._to_tensor((eta_prev.reshape(-1, 1) * e_a_t
                                     * zxa_ind).mean(0))
        obs_freqs = self._to_tensor(zxa_ind.mean(0))

        # setup q network
        q_net = self.q_net_class(**self.q_net_args, x_len=zx_len, a_len=a_len,
                                 num_a=self.num_a)
        optimizer = torch.optim.LBFGS(q_net.parameters())

        def closure():
            optimizer.zero_grad()
            q_net_vals = q_net(zxa_vals_torch).flatten()
            rho = b @ q_net_vals - c
            l0, l1, l2 = self.lmbda
            reg_0 = l0 * (q_net_vals ** 2) @ obs_freqs
            reg_1 = l1 * (freqs @ q_net_vals - self.num_a) ** 2
            reg_2 = l2 * (reg_freqs @ q_net_vals - 1.0) ** 2
            loss = rho.T @ k @ rho + reg_0 + reg_1 + reg_2
            loss.backward()
            return loss
        optimizer.step(closure)

        q_net_vals = q_net(zxa_vals_torch).flatten()
        rho = b @ q_net_vals - c
        loss = float(rho.T @ k @ rho)

        q_vals = q_net(zxa_vals_torch).detach().flatten()
        # print(np.array(sorted(torch_to_np(q_vals))))
        mean_q = float(freqs @ q_vals)
        std_q = float(freqs @ ((q_vals - mean_q) ** 2))
        # print("mean q:", mean_q, "std q:", std_q, "loss:", loss)
        # print(float(reg_freqs @ q_vals))

        norm = float(reg_freqs @ q_vals)
        self.q_net = q_net
        return partial(self.q_func, q_net=q_net, norm=norm)

    def q_func(self, z_t, x_t, a_t, q_net, norm):
        if x_t is None:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, a_t))
            zxa_vals_embed = np.concatenate([self.embed_z(zxa_vals[:, 0]),
                                             self.embed_a(zxa_vals[:, 1])],
                                            axis=1)
        else:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, x_t, a_t))
            zxa_vals_embed = np.concatenate([self.embed_z(zxa_vals[:, 0]),
                                             self.embed_x(zxa_vals[:, 1]),
                                             self.embed_a(zxa_vals[:, 2])],
                                            axis=1)

        q_net_vals = q_net(self._to_tensor(zxa_vals_embed)) / norm
        return torch_to_np(q_net_vals)[zxa_codes]

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class DiscreteMMRHEstimation(AbstractHEstimator):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, num_a,
                 zxa_sq_dist, wxa_sq_dist, h_net_class, h_net_args,
                 f_kernel_class, f_kernel_args, lmbda, cuda=False, device=None):
        AbstractHEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_x=embed_x, embed_a=embed_a,
                                    num_a=num_a, zxa_sq_dist=zxa_sq_dist,
                                    wxa_sq_dist=wxa_sq_dist)

        self.h_net_class = h_net_class
        self.h_net_args = h_net_args
        self.f_kernel = f_kernel_class(sq_dist_func=zxa_sq_dist,
                                       **f_kernel_args)

        self.lmbda = lmbda
        self.cuda = cuda
        self.device = device

        self.h_net = None

    def fit(self, eta_prev, e_t, y_t, z_t, w_t, x_t, a_t, dfr_min, dfr_max):
        eta_prev = eta_prev / eta_prev.mean()
        # compute codes for each of z, w, x
        n = len(w_t)
        if x_t is None:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, a_t))
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, a_t))
        else:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, x_t, a_t))
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, x_t, a_t))

        zxa_vals_embed, wxa_vals_embed, a_len = compute_triplet_embeddings(
            zxa_vals, wxa_vals, z_t, w_t, x_t, self.num_a,
            self.embed_z, self.embed_w, self.embed_x, self.embed_a)
        wx_len = wxa_vals_embed.shape[-1] - a_len
        wxa_vals_torch = self._to_tensor(wxa_vals_embed)

        # train kernel hyper-parameters
        self.f_kernel.train(zxa_vals)

        # set up basic matrices / data structures that are constant for each
        # iteration
        k_reg = 1e-1 * np.eye(len(zxa_vals))
        k = self._to_tensor(self.f_kernel(zxa_vals, zxa_vals) + k_reg)

        zxa_ind = one_hot_embed(zxa_codes, num_categories=len(zxa_vals))
        wxa_ind = one_hot_embed(wxa_codes, num_categories=len(wxa_vals))
        b = self._to_tensor((1/n) * (zxa_ind.T @ wxa_ind))
        target = y_t * (e_t == a_t).reshape(-1, 1)
        c = self._to_tensor((target * zxa_ind).mean(0))

        freqs = self._to_tensor((eta_prev.reshape(-1, 1) * wxa_ind).mean(0))
        obs_freqs = self._to_tensor(wxa_ind.mean(0))

        # setup q network
        h_net = self.h_net_class(**self.h_net_args, x_len=wx_len, a_len=a_len,
                                 num_a=self.num_a)
        optimizer = torch.optim.LBFGS(h_net.parameters())

        def closure():
            optimizer.zero_grad()
            h_net_vals = h_net(wxa_vals_torch).flatten()
            rho = b @ h_net_vals - c
            l0, l1, l2 = self.lmbda
            reg_0 = l0 * (h_net_vals ** 2) @ obs_freqs
            reg_1 = l1 * (h_net_vals @ freqs - target.mean()) ** 2
            h_net_excess = (F.relu(h_net_vals - dfr_max)
                            + F.relu(dfr_min - h_net_vals)) ** 2
            reg_2 = l2 * h_net_excess @ obs_freqs
            loss = rho.T @ k @ rho + reg_0 + reg_1 + reg_2
            loss.backward()
            return loss

        optimizer.step(closure)

        h_net_vals = h_net(wxa_vals_torch).flatten()
        rho = b @ h_net_vals - c
        loss = float(rho.T @ k @ rho)

        h_vals = h_net(wxa_vals_torch).detach().flatten()
        # print(np.array(sorted(torch_to_np(h_vals))))
        mean_h = float(freqs @ h_vals)
        std_h = float(freqs @ ((h_vals - mean_h) ** 2))
        # print("mean h:", mean_h, "std h:", std_h, "loss:", loss)

        norm = mean_h
        self.h_net = h_net
        return partial(self.h_func, h_net=h_net, norm=norm)

    def h_func(self, w_t, x_t, a_t, h_net, norm):
        if x_t is None:
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, a_t))
            wxa_vals_embed = np.concatenate([self.embed_w(wxa_vals[:, 0]),
                                             self.embed_a(wxa_vals[:, 1])],
                                            axis=1)
        else:
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, x_t, a_t))
            wxa_vals_embed = np.concatenate([self.embed_w(wxa_vals[:, 0]),
                                             self.embed_x(wxa_vals[:, 1]),
                                             self.embed_a(wxa_vals[:, 2])],
                                            axis=1)

        h_net_vals = h_net(self._to_tensor(wxa_vals_embed)) / norm
        return torch_to_np(h_net_vals)[wxa_codes]

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)

from functools import partial
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

from nuisance_estimation.abstract_nuisance import AbstractNuisance
from nuisance_estimation.discrete_mmr_nuisance_estimation import \
    DiscreteMMRNuisanceEstimation, DiscreteMMRQEstimation, \
    DiscreteMMRHEstimation
from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.np_utils import one_hot_embed
from utils.oadam import OAdam
from utils.torch_utils import torch_to_np, np_to_tensor, BatchIter


class JointDoubleNeuralNuisanceEstimation(AbstractNuisance):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, zxa_sq_dist,
                 wxa_sq_dist, horizon, gamma, num_a, q_net_class, q_net_args,
                 g_net_class, g_net_args, h_net_class, h_net_args,
                 f_net_class, f_net_args, eta_lmbda, y_lmbda,
                 q_lr, g_lr, h_lr, f_lr,  batch_size, num_epochs_per_stage,
                 num_epochs_final, eval_freq=None, pretrain=False,
                 pretrain_kernel_class=None, pretrain_kernel_args=None,
                 pretrain_q_lmbda=None, pretrain_h_lmbda=None,
                 cuda=False, device=None):
        self. q_net_class = q_net_class
        self. q_net_args = q_net_args
        self. g_net_class = g_net_class
        self. g_net_args = g_net_args
        self. h_net_class = h_net_class
        self. h_net_args = h_net_args
        self. f_net_class = f_net_class
        self. f_net_args = f_net_args

        self.batch_size = batch_size
        self.num_epochs_per_stage = num_epochs_per_stage
        self.num_epochs_final = num_epochs_final
        self.eval_freq = eval_freq

        self.q_list = []
        self.h_list = []

        self.eta_lmbda = eta_lmbda
        self.y_lmbda = y_lmbda

        self.pretrain = pretrain
        self.pretrain_kernel_class = pretrain_kernel_class
        self.pretrain_kernel_args = pretrain_kernel_args
        self.pretrain_q_lmbda = pretrain_q_lmbda
        self.pretrain_h_lmbda = pretrain_h_lmbda

        self.q_lr = q_lr
        self.g_lr = g_lr
        self.h_lr = h_lr
        self.f_lr = f_lr

        self.cuda = cuda
        self.device = device

        AbstractNuisance.__init__(
            self, horizon=horizon, gamma=gamma, num_a=num_a, embed_z=embed_z,
            embed_x=embed_x, embed_w=embed_w, embed_a=embed_a,
            zxa_sq_dist=zxa_sq_dist, wxa_sq_dist=wxa_sq_dist)

    def fit(self, pci_dataset):
        n = pci_dataset.get_n()
        zxa_list = []
        wxa_list = []
        wxa_a_lists = []
        e_a_list = []
        r_list = []
        g_list = []
        f_list = []
        q_optim_list = []
        h_optim_list = []
        g_optim_list = []
        f_optim_list = []

        if self.pretrain:
            assert self.pretrain_kernel_class is not None
            assert self.pretrain_kernel_args is not None
            assert self.pretrain_q_lmbda is not None
            assert self.pretrain_h_lmbda is not None
            print("pretraining START")
            self.pretrain_q_h(pci_dataset)
            print("pretraining DONE")

        # first set up embeddings, networks, and optimizers
        for t in range(self.horizon):
            z_t = pci_dataset.get_z_t(t)
            w_t = pci_dataset.get_w_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            e_t = pci_dataset.get_e_t(t)
            e_a_t = self._to_tensor((a_t == e_t).astype(int)).view(-1, 1)
            e_a_list.append(e_a_t)
            r_list.append(self._to_tensor(pci_dataset.get_r_t(t)).view(-1, 1))
            zxa_embed, zx_len, a_len = self.embed_zxa(z_t, x_t, a_t)
            wxa_embed, wx_len, a_len_2 = self.embed_wxa(w_t, x_t, a_t)
            assert a_len == a_len_2

            if x_t is None:
                w_t_embed = self.embed_w(w_t)
                wxa_a_list = []
                for a in range(self.num_a):
                    a_embed = self.embed_a(np.array([a for _ in range(n)]))
                    wxa_a_embed = np.concatenate([w_t_embed, a_embed], axis=1)
                    wxa_a_list.append(self._to_tensor(wxa_a_embed))
                wxa_a_lists.append(wxa_a_list)
            else:
                w_t_embed = self.embed_w(w_t)
                x_t_embed = self.embed_x(x_t)
                wxa_a_list = []
                for a in range(self.num_a):
                    a_embed = self.embed_a(np.array([a for _ in range(n)]))
                    wxa_a_embed = np.concatenate(
                        [w_t_embed, x_t_embed, a_embed], axis=1)
                    wxa_a_list.append(self._to_tensor(wxa_a_embed))
                wxa_a_lists.append(wxa_a_list)

            zxa_list.append(zxa_embed)
            wxa_list.append(wxa_embed)
            g_t = self.g_net_class(
                **self.g_net_args, x_len=wx_len, a_len=a_len, num_a=self.num_a)
            f_t = self.f_net_class(
                **self.f_net_args, x_len=zx_len, a_len=a_len, num_a=self.num_a)
            g_list.append(g_t)
            f_list.append(f_t)
            if self.pretrain:
                q_t = self.q_list[t]
                h_t = self.h_list[t]
            else:
                q_t = self.q_net_class(
                    **self.q_net_args, x_len=zx_len, a_len=a_len,
                    num_a=self.num_a)
                h_t = self.h_net_class(
                    **self.h_net_args, x_len=wx_len, a_len=a_len,
                    num_a=self.num_a)
                self.q_list.append(q_t)
                self.h_list.append(h_t)

            dis = self.gamma ** t
            q_optim_list.append(OAdam(q_t.parameters(), lr=dis*self.q_lr))
            g_optim_list.append(OAdam(g_t.parameters(), lr=dis*self.g_lr))
            h_optim_list.append(OAdam(h_t.parameters(), lr=dis*self.h_lr))
            f_optim_list.append(OAdam(f_t.parameters(), lr=dis*self.f_lr))
            qh_optim_list = q_optim_list + h_optim_list
            gf_optim_list = g_optim_list + f_optim_list

        batch_iter = BatchIter(pci_dataset.get_n(), self.batch_size)
        max_num_parts = self.horizon * 2
        num_epochs = (self.num_epochs_per_stage * max_num_parts
                      + self.num_epochs_final)
        for epoch in range(num_epochs):
            num_parts = min(epoch // self.num_epochs_per_stage + 1,
                            max_num_parts)
            for batch_idx in batch_iter:
                obj, _, _ = self.compute_game_obj(
                    g_list, f_list, wxa_list, wxa_a_lists, zxa_list, e_a_list,
                    r_list, num_parts=num_parts, batch_idx=batch_idx,
                    lmbda_reg=True)

                # update all networks
                for optim in qh_optim_list:
                    optim.zero_grad()
                    obj.backward(retain_graph=True)
                    optim.step()
                for optim in gf_optim_list[:-1]:
                    optim.zero_grad()
                    (-1.0 * obj).backward(retain_graph=True)
                    optim.step()
                gf_optim_list[-1].zero_grad()
                (-1.0 * obj).backward()
                gf_optim_list[-1].step()

            if (self.eval_freq is not None) and (epoch % self.eval_freq == 0):
                eval_obj, q_obj_list, h_obj_list = self.compute_game_obj(
                    g_list, f_list, wxa_list, wxa_a_lists, zxa_list, e_a_list,
                    r_list, num_parts=num_parts,
                    batch_idx=np.arange(n), lmbda_reg=False)
                print("epoch = %d, obj = %f, q_obj = %r, h_obj = %r"
                      % (epoch, eval_obj, q_obj_list, h_obj_list))

    def compute_game_obj(self, g_list, f_list, wxa_list, wxa_a_lists,
                         zxa_list, e_a_list, r_list, batch_idx, num_parts,
                         lmbda_reg=True):

        # construct eta vectors for batch
        eta = 1.0
        batch_size = len(batch_idx)
        eta_prev_list = [torch.ones(batch_size, 1)]
        q_list = []
        for t in range(self.horizon):
            q = self.q_list[t](zxa_list[t][batch_idx])
            q_list.append(q)
            eta = eta * q * e_a_list[t][batch_idx]
            if t < self.horizon - 1:
                eta_prev_list.append(eta)

        # construct y vectors from batch
        y_list = []
        h_list = []
        h_a_lists = []
        t_rev = list(range(self.horizon))[::-1]
        for t in t_rev:
            r = r_list[t][batch_idx]
            if t == self.horizon - 1:
                y_list.append(r)
            else:
                target_prev = e_a_list[t][batch_idx] * y_list[-1]
                y_residual = q_list[t + 1] * (target_prev - h_list[-1])
                for a in range(self.num_a):
                    y_residual = y_residual + h_a_lists[-1][a]
                y_list.append(r + self.gamma * y_residual)

            h = self.h_list[t](wxa_list[t][batch_idx])
            h_list.append(h)
            h_a_list = []
            for a in range(self.num_a):
                h_a = self.h_list[t](wxa_a_lists[t][a][batch_idx])
                h_a_list.append(h_a)
            h_a_lists.append(h_a_list)
        y_list.reverse()
        h_list.reverse()
        h_a_lists.reverse()

        # construct min-max objective
        obj_parts = []
        reg_parts = []
        q_obj_list = []
        h_obj_list = []
        part_i = 0
        for t in range(self.horizon):
            part_i += 1
            if part_i > num_parts:
                break

            eta_prev = eta_prev_list[t]

            # compute q/g objective
            g = g_list[t](wxa_list[t][batch_idx])
            q_vec = eta_prev * q_list[t]
            g_vec = 0
            for a in range(self.num_a):
                g_vec = g_vec + g_list[t](wxa_a_lists[t][a][batch_idx])
            q_obj = (g * q_vec - g_vec * eta_prev).mean()
            q_reg = g * q_vec.detach() - g_vec * eta_prev.detach()
            if lmbda_reg:
                eta_reg_0 = self.eta_lmbda[0] * ((eta_prev - 1.0) ** 2).mean()
                eta_reg_1 = self.eta_lmbda[1] * (eta_prev.mean() - 1.0) ** 2
                q_obj = q_obj + eta_reg_0 + eta_reg_1
            obj_parts.append(q_obj)
            reg_parts.append(q_reg)
            q_obj_list.append(float(q_obj - 0.25 * (q_reg ** 2).mean()))

        for t in t_rev:
            part_i += 1
            if part_i > num_parts:
                break

            # compute h/f objective
            f = f_list[t](zxa_list[t][batch_idx])
            h_vec = h_list[t] - y_list[t] * e_a_list[t][batch_idx]
            h_obj = (f * eta_prev * h_vec).mean()
            h_reg = f * (eta_prev * h_vec).detach()
            if lmbda_reg:
                target = (eta_prev * y_list[t] * e_a_list[t][batch_idx]).mean()
                eta_h = eta_prev * h_list[t]
                y_reg_0 = self.y_lmbda[0] * ((eta_h - target) ** 2).mean()
                y_reg_1 = self.y_lmbda[1] * (eta_h.mean() - target) ** 2
                h_obj = h_obj + y_reg_0 + y_reg_1
            obj_parts.append(h_obj)
            reg_parts.append(h_reg)
            h_obj_list.append(float(h_obj - 0.25 * (h_reg ** 2).mean()))

        obj = sum(obj_parts)
        reg = sum(reg_parts)
        obj = obj - 0.25 * (reg ** 2).mean()
        return obj, np.array(q_obj_list), np.array(h_obj_list)

    def q_t(self, t, z_t, x_t, a_t):
        zxa_embed, _, _ = self.embed_zxa(z_t, x_t, a_t)
        return torch_to_np(self.q_list[t](zxa_embed))

    def h_t(self, t, w_t, x_t, a_t):
        wxa_embed, _, _ = self.embed_wxa(w_t, x_t, a_t)
        return torch_to_np(self.h_list[t](wxa_embed))

    def embed_zxa(self, z_t, x_t, a_t):
        if x_t is None:
            z_t_embed = self.embed_z(z_t)
            a_t_embed = self.embed_a(a_t)
            zxa_embed = np.concatenate([z_t_embed, a_t_embed], axis=1)
            zx_len = z_t_embed.shape[1]
            a_len = a_t_embed.shape[1]
        else:
            z_t_embed = self.embed_z(z_t)
            x_t_embed = self.embed_x(x_t)
            a_t_embed = self.embed_a(a_t)
            zxa_embed = np.concatenate([z_t_embed, x_t_embed, a_t_embed],
                                       axis=1)
            zx_len = z_t_embed.shape[1] + x_t_embed.shape[1]
            a_len = a_t_embed.shape[1]

        return self._to_tensor(zxa_embed), zx_len, a_len

    def embed_wxa(self, w_t, x_t, a_t):
        if x_t is None:
            w_t_embed = self.embed_w(w_t)
            a_t_embed = self.embed_a(a_t)
            wxa_embed = np.concatenate([w_t_embed, a_t_embed], axis=1)
            wx_len = w_t_embed.shape[1]
            a_len = a_t_embed.shape[1]
        else:
            w_t_embed = self.embed_w(w_t)
            x_t_embed = self.embed_x(x_t)
            a_t_embed = self.embed_a(a_t)
            wxa_embed = np.concatenate([w_t_embed, x_t_embed, a_t_embed],
                                       axis=1)
            wx_len = w_t_embed.shape[1] + x_t_embed.shape[1]
            a_len = a_t_embed.shape[1]

        return self._to_tensor(wxa_embed), wx_len, a_len

    def pretrain_q_h(self, pci_dataset):
        nu_list = []
        eta_list = []
        q_list = []
        min_r_list = []
        max_r_list = []
        n = pci_dataset.get_n()
        t_range = list(range(self.horizon))

        # first, fit the q functions one by one
        assert pci_dataset.get_horizon() == self.horizon
        for t in t_range:
            z_t = pci_dataset.get_z_t(t)
            w_t = pci_dataset.get_w_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            e_t = pci_dataset.get_e_t(t)
            r_t = pci_dataset.get_r_t(t)
            if t == 0:
                eta_prev = np.ones((n, 1))
            else:
                eta_prev = eta_list[-1]

            # fit q
            q_estimator = DiscreteMMRQEstimation(
                embed_z=self.embed_z, embed_w=self.embed_w,
                embed_x=self.embed_x, embed_a=self.embed_a, num_a=self.num_a,
                zxa_sq_dist=self.zxa_sq_dist, wxa_sq_dist=self.wxa_sq_dist,
                q_net_class=self.q_net_class, q_net_args=self.q_net_args,
                g_kernel_class=self.pretrain_kernel_class,
                g_kernel_args=self.pretrain_kernel_args,
                lmbda=self.pretrain_q_lmbda)

            q_t = q_estimator.fit(eta_prev, z_t, w_t, x_t, a_t, e_t)
            self.q_list.append(q_estimator.q_net)

            # calculate next nu and eta
            q_t_array = q_t(z_t, x_t, a_t)
            nu_list.append(eta_prev * q_t_array)
            eta_list.append(nu_list[-1] * (e_t == a_t).reshape(-1, 1))
            q_list.append(q_t_array)

            min_r_list.append(float(r_t.min()))
            max_r_list.append(float(r_t.max()))

        # next, fit the h functions backwards one by one
        reverse_h_list = []
        dfr_min = 0
        dfr_max = 0
        y_t_prev = np.zeros((n, 1))
        for t in t_range[::-1]:
            z_t = pci_dataset.get_z_t(t)
            w_t = pci_dataset.get_w_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            r_t = pci_dataset.get_r_t(t)
            e_t = pci_dataset.get_e_t(t)
            if t == 0:
                eta_prev = np.ones((n, 1))
            else:
                eta_prev = eta_list[-1]

            y_t = r_t.reshape(-1, 1) + self.gamma * y_t_prev
            dfr_min = min_r_list[t] + self.gamma * dfr_min
            dfr_max = max_r_list[t] + self.gamma * dfr_max

            # fit h
            h_estimator = DiscreteMMRHEstimation(
                embed_z=self.embed_z, embed_w=self.embed_w,
                embed_x=self.embed_x, embed_a=self.embed_a, num_a=self.num_a,
                zxa_sq_dist=self.zxa_sq_dist, wxa_sq_dist=self.wxa_sq_dist,
                h_net_class=self.h_net_class, h_net_args=self.h_net_args,
                f_kernel_class=self.pretrain_kernel_class,
                f_kernel_args=self.pretrain_kernel_args,
                lmbda=self.pretrain_h_lmbda)

            h_t = h_estimator.fit(eta_prev, e_t, y_t, z_t, w_t, x_t, a_t,
                                  dfr_min, dfr_max)
            self.h_list.append(h_estimator.h_net)

            # work out remainder for next calculation
            h_t_sum = np.zeros((n, 1))
            for a in range(self.num_a):
                a_const = np.array([a for _ in range(n)])
                h_t_sum = h_t_sum + h_t(w_t, x_t, a_const)

            y_t_prev = q_list[t] * ((a_t == e_t).reshape(-1, 1) * y_t
                                    - h_t(w_t, x_t, a_t)) + h_t_sum

        self.h_list.reverse()

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


def hash_np(val):
    if isinstance(val, np.ndarray):
        return tuple(val.flatten())
    else:
        return val

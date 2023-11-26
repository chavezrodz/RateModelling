import numpy as np
import torch


def log_to_lin(x, x_mean, norm):
    x = (torch.log10(x) - x_mean) / norm
    return x


def lin_to_log(x, x_mean, norm):
    x = x * norm + x_mean
    return 10**x


def get_mean_norm(x, log=True):
    if log:
        x = np.log10(x)
    return x.mean(), (x.max() - x.min()) / 2


def get_consts_dict_modelling(x):
    t_mn = get_mean_norm(torch.unique(x[:, -2]))
    # t_mn = get_mean_norm(torch.unique(x[:, -2]), log=False)
    T_mn = get_mean_norm(torch.unique(x[:, -3]), log=False)
    p_mn = get_mean_norm(torch.unique(x[:, -5]))

    consts = torch.zeros(3, 2)
    consts[0, 0], consts[0, 1] = p_mn
    consts[1, 0], consts[1, 1] = T_mn
    consts[2, 0], consts[2, 1] = t_mn
    return consts


def get_consts_dict_integrating(x):
    t_mn = get_mean_norm(torch.unique(x[:, -2]))
    T_mn = get_mean_norm(torch.unique(x[:, -3]), log=False)
    p_mn = get_mean_norm(torch.unique(x[:, -4]))

    consts = torch.zeros(3, 2)
    consts[0, 0], consts[0, 1] = p_mn
    consts[1, 0], consts[1, 1] = T_mn
    consts[2, 0], consts[2, 1] = t_mn
    return consts


def normalize_vector_modelling(x, consts):
    """
    Proceed in reverse order: t, T, K, P
    """
    # x[:, -1] = (x[:, -1] - consts[-1, 0])/consts[-1, 1]
    x[:, -1] = log_to_lin(x[:, -1], consts[-1, 0], consts[-1, 1])
    x[:, -2] = (x[:, -2] - consts[-2, 0]) / consts[-2, 1]
    x[:, -3] = x[:, -3] / x[:, -4]
    x[:, -4] = log_to_lin(x[:, -4], consts[-3, 0], consts[-3, 1])
    return x


def normalize_vector_integrating(x, consts):
    """
    Proceed in reverse order: t, T, K, P
    """

    x[:, -1] = log_to_lin(x[:, -1], consts[-1, 0], consts[-1, 1])
    x[:, -2] = (x[:, -2] - consts[-2, 0]) / consts[-2, 1]
    x[:, -3] = log_to_lin(x[:, -3], consts[-3, 0], consts[-3, 1])
    return x

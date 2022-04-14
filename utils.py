import numpy as np


def get_mean_norm(x, log=True):
    if log:
        x = np.log10(x)
    return x.mean(), (x.max() - x.min())/2


def log_to_lin(x, x_mean, norm):
    x = (np.log10(x) - x_mean)/norm
    return x


def lin_to_log(x, x_mean, norm):
    x = x*norm + x_mean
    return 10**x


def normalize_vector(x, p_m, p_n, T_m, T_n, t_m, t_n):
    """
    Proceed in reverse order: t, T, K, P
    """

    x[:, -1] = log_to_lin(x[:, -1], t_m, t_n)
    x[:, -2] = (x[:, -2] - T_m)/T_n
    x[:, -3] = x[:, -3]/x[:, -4]
    x[:, -4] = log_to_lin(x[:, -4], p_m, p_n)
    return x


def pc_err(pred, y):
    return ((pred-y)/y).abs().mean()

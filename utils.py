import pandas as pd
import numpy as np
import torch
import os


def generate_test_data(P, K, T, N=101):
    t = torch.linspace(0.05, 20, N)
    P = torch.ones(N)*P
    K = torch.ones(N)*K
    T = torch.ones(N)*T
    return torch.stack([P, K, T, t]).T


def get_mean_norm(x, log=True):
    if log:
        x = np.log10(x)
    return x.mean(), (x.max() - x.min())/2


def get_consts_dict(x):
    t_mn = get_mean_norm(np.unique(x[:, -2]))
    T_mn = get_mean_norm(np.unique(x[:, -3]), log=False)
    p_mn = get_mean_norm(np.unique(x[:, -5]))
    cons_dict = {
        't_m': t_mn[0],
        't_n': t_mn[1],
        'T_m': T_mn[0],
        'T_n': T_mn[1],
        'p_m': p_mn[0],
        'p_n': p_mn[1],
    }
    return cons_dict


def log_to_lin(x, x_mean, norm):
    x = (np.log10(x) - x_mean)/norm
    return x


def lin_to_log(x, x_mean, norm):
    x = x*norm + x_mean
    return 10**x


def normalize_vector(x, consts):
    """
    Proceed in reverse order: t, T, K, P
    """

    x[:, -1] = log_to_lin(x[:, -1], consts['t_m'], consts['t_n'])
    x[:, -2] = (x[:, -2] - consts['T_m'])/consts['T_n']
    x[:, -3] = x[:, -3]/x[:, -4]
    x[:, -4] = log_to_lin(x[:, -4], consts['p_m'], consts['p_n'])
    return x


def pc_err(pred, y):
    return ((pred-y)/y).abs().mean()


def make_file_prefix(args):
    file_prefix = 'M_' + str(args.method)
    file_prefix += '_n_layers_' + str(args.n_layers)
    file_prefix += '_hid_dim_' + str(args.hidden_dim)
    file_prefix += '_val_pc_err='
    return file_prefix


def make_checkpt_dir(args):
    dir = os.path.join(
        args.results_dir,
        "saved_models",
        'Method_'+str(args.method)
        )
    return dir

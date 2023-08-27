import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def closest_value(df, col, val):
    uniques = df[col].unique()
    close_idx = np.argmin(np.abs(uniques - val))
    return uniques[close_idx]


def get_closest_values(df, P, K, T):
    p_closest = closest_value(df, 'P', P)
    T_closest = closest_value(df, 'T', T)
    subset = df[(df['P'] == p_closest) & (df['T'] == T_closest)]
    k_closest = closest_value(subset, 'K', K)
    subset = subset.loc[df['K'] == k_closest]
    return subset


def generate_test_data(P, K, T, N=50):
    out = torch.zeros(N, 4)
    out[:, 0] = P
    out[:, 1] = K
    out[:, 2] = T
    out[:, 3] = torch.linspace(0, 20, N)
    return out


def make_file_prefix(args):
    file_prefix = 'M_' + str(args.method)
    file_prefix += '_n_layers_' + str(args.n_layers)
    file_prefix += '_hid_dim_' + str(args.hidden_dim)
    file_prefix += '_val_pc_err='
    return file_prefix


def make_checkpt_dir(args):
    dir = os.path.join(
        args.results_dir,
        args.proj_dir,
        "saved_models",
        'Method_'+str(args.method)
        )
    return dir

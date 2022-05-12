import pandas as pd
import numpy as np
import torch
import os


def log_to_lin(x, x_mean, norm):
    x = (torch.log10(x) - x_mean)/norm
    return x


def lin_to_log(x, x_mean, norm):
    x = x*norm + x_mean
    return 10**x


def get_mean_norm(x, log=True):
    if log:
        x = np.log10(x)
    return x.mean(), (x.max() - x.min())/2


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
    out[:, 0] = K
    out[:, 0] = T
    out[:, 4] = np.linspace(0, 20, N)
    return out


def get_consts_dict(x):
    t_mn = get_mean_norm(torch.unique(x[:, -2]))
    T_mn = get_mean_norm(torch.unique(x[:, -3]), log=False)
    p_mn = get_mean_norm(torch.unique(x[:, -5]))

    consts = torch.zeros(3, 2)
    consts[0, 0], consts[0, 1] = p_mn
    consts[1, 0], consts[1, 1] = T_mn
    consts[2, 0], consts[2, 1] = t_mn
    return consts


def normalize_vector(x, consts):
    """
    Proceed in reverse order: t, T, K, P
    """

    x[:, -1] = log_to_lin(x[:, -1], consts[-1, 0], consts[-1, 1])
    x[:, -2] = (x[:, -2] - consts[-2, 0])/consts[-2, 1]
    x[:, -3] = x[:, -3]/x[:, -4]
    x[:, -4] = log_to_lin(x[:, -4], consts[-3, 0], consts[-3, 1])
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


def load_df(datapath, datafile, which):
    if which == 'both':
        df1 = pd.read_csv(os.path.join(datapath, 'logspaced', datafile))
        df2 = pd.read_csv(os.path.join(datapath, 'linspaced', datafile))
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(datapath, which, datafile))

    df = df.drop_duplicates()

    ts = df['t'].unique()
    t_10 = ts[np.argmin(np.abs(ts - 10))]
    t_max = np.max(ts)

    # print(len(df))
    df_10 = df[df['t'] == t_10].sort_values(by=['P', 'K', 'T'], ascending=True)
    df_max = df[df['t'] == t_max].sort_values(by=['P', 'K', 'T'], ascending=True)

    # print(np.array_equal(df_10.values[:, :-2], df_max.values[:, :-2]))
    g10 = df_10['gamma'].values
    gmax = df_max['gamma'].values

    err = np.abs((g10 - gmax) / g10) * 100

    df_10['pc_err'] = err
    df_10 = df_10.drop(['gamma', 't'], axis=1)
    # print(df_10)
    idx = np.where(err > 1000)[0]
    df_th = df_10.iloc[idx].sort_values(by='pc_err', ascending=True)
    # print(len(err), len(idx))

    # print(df_th)
    # df = df[df['t'] < 10]
    return df
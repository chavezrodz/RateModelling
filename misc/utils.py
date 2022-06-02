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
    out[:, 0] = K
    out[:, 0] = T
    out[:, 4] = np.linspace(0, 20, N)
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


def load_df(datapath, datafile, which, verbose=False):
    if which == 'both':
        df1 = pd.read_csv(os.path.join(datapath, 'logspaced', datafile))
        df2 = pd.read_csv(os.path.join(datapath, 'linspaced', datafile))
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(datapath, which, datafile))

    df = df.drop_duplicates()
    init_len = len(df.copy())

    ts = df['t'].unique()
    t_10 = ts[np.argmin(np.abs(ts - 10))]
    t_max = np.max(ts)

    # print(len(df))
    df_10 = df[df['t'] == t_10].sort_values(
        by=['P', 'K', 'T'], ascending=True
        )
    df_max = df[df['t'] == t_max].sort_values(
        by=['P', 'K', 'T'], ascending=True
        )

    assert np.array_equal(df_10.values[:, :-2], df_max.values[:, :-2])

    g10 = df_10['gamma'].values
    gmax = df_max['gamma'].values

    err = np.abs((g10 - gmax) / g10) * 100

    df_10['pc_err'] = err
    df_10 = df_10.drop(['gamma', 't'], axis=1)

    # idx = np.where( (300 > err) & (err > 200))[0]
    idx = np.where(err > 200)[0]
    df_th = df_10.iloc[idx].sort_values(by='pc_err', ascending=True)

    indices = []
    for line in df_th.values:
        P, K, T = line[1], line[2], line[3]
        idx = df.loc[((df['P'] == P) & (df['K'] == K) & (df['T'] == T)), :]
        indices.extend(idx.index.values)
        if verbose:
            print(f'P:{P}, K:{K}, T:{T}')
            subset = df.loc[idx.values].sort_values(by='t', ascending=True)
            plt.plot(subset['t'], subset['gamma']/subset['gamma'].max())

    indices_bad = pd.Index(indices)
    indices_good = df.index.difference(indices_bad)
    df = df.loc[indices_good]
    if verbose:
        print(f'Number of corrupt lines: {len(idx)}/{len(err)}')
        print(
            f'Total points used {len(df)}/{init_len} ({len(df)/init_len*100}%)'
            )
        plt.show()
    return df

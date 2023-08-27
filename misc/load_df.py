import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_df(datapath, method, which, verbose=False):
    datafile = 'method_'+str(method)+'.csv'
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

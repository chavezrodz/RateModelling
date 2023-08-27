
from pytorch_lightning import Trainer
import numpy as np
import torch
import os
import scipy.integrate as integrate
from misc.Datamodule import DataModule
from misc.load_df import load_df
from misc.utils import generate_test_data, get_closest_values


def compare_with_data(P, K, T, datapath, method, model, include_data):
    K *= P
    data_gam = None
    if include_data:
        df = load_df(datapath, method, 'both')
        df = get_closest_values(df, P, K, T)
        df = df.sort_values(by='t', ascending=True)

        data_gam = df['gamma']
        t = df['t'].values
        X = df[['P', 'K', 'T', 't']].values
        P = df['P'].iloc[0]
        K = df['K'].iloc[0]
        T = df['T'].iloc[0]

        X = torch.tensor(X).type(torch.float)

    else:
        X = generate_test_data(P, K, T)
        t = X[:, -1].clone()

    pred = model(X).detach().squeeze().numpy()
    return t, pred, data_gam


def compare_k_preds(P, T, t, datapath, method, model,
                    a, b, N=50, which='both'):
    # Data
    df = load_df(datapath, method, which)

    p_values = np.sort(df['P'].unique())
    T_values = np.sort(df['T'].unique())
    t_values = np.sort(df['t'].unique())

    p_idx = np.argmin(np.abs(P - p_values))
    T_idx = np.argmin(np.abs(T - T_values))
    t_idx = np.argmin(np.abs(t - t_values))

    subset = df[(df['P'] == p_values[p_idx]) &
                (df['T'] == T_values[T_idx]) &
                (df['t'] == t_values[t_idx])]

    subset = subset.drop(['M'], axis=1).sort_values('K')

    # Model
    K = torch.linspace(a, b, N) * P
    X = torch.zeros(N, 4)
    X[:, 0] = P
    X[:, 1] = K
    X[:, 2] = T
    X[:, 3] = t
    pred = model(X).detach().squeeze()
    return (K, pred), (subset['K'], subset['gamma'])


def k_integral(P, T, t, datapath, method, model, a, b, N=50, which='both'):
    # Data
    df = load_df(datapath, method, which)

    p_values = np.sort(df['P'].unique())
    T_values = np.sort(df['T'].unique())
    t_values = np.sort(df['t'].unique())

    p_idx = np.argmin(np.abs(P - p_values))
    T_idx = np.argmin(np.abs(T - T_values))
    t_idx = np.argmin(np.abs(t - t_values))

    subset = df[(df['P'] == p_values[p_idx]) &
                (df['T'] == T_values[T_idx]) &
                (df['t'] == t_values[t_idx])]

    subset = subset.drop(['M'], axis=1).sort_values('K')

    x, y = subset['K'], subset['gamma']

    # romb_integral = integrate.romb(y, np.diff(x).mean())
    data_integral = integrate.simpson(y, x=x)
    # Model

    def func(k):
        k = torch.tensor(k)
        X = torch.zeros(len(k), 4)
        X[:, 0] = P
        X[:, 1] = k
        X[:, 2] = T
        X[:, 3] = t
        return np.array(model(X).detach().squeeze())

    return data_integral, integrate.quadrature(func, a*P, b*P)

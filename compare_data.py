import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from MLP import MLP
import os


def generate_test_data(P, K, T, N=50):
    t = torch.linspace(0, 20, N)
    P = torch.ones(N)*P
    K = torch.ones(N)*K
    T = torch.ones(N)*T

    return torch.stack([P, K, T, t]).T


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


def compare_with_data(P, K, T, dataset, model, hidden_dim, n_layers):
    df = pd.read_csv(os.path.join('datasets', dataset))
    df = df[df['gamma'] > 0]
    df = get_closest_values(df, P, K, T)

    data_gam = df['gamma']
    t = df['t'].values
    X = df[['P', 'K', 'T', 't']].values
    X = torch.tensor(X).type(torch.float)

    pred = model(X).detach().squeeze()
    print(f"""
    Using M = {df['M'].iloc[0]}
        P = {df['P'].iloc[0]}
        K = {df['K'].iloc[0]}
        T = {df['T'].iloc[0]}
    Error: {((pred-data_gam)/data_gam).abs().mean()*100:.2f}%
    """)
    # {pred.shape}{data_gam.shape}

    plt.plot(t, data_gam, label='data')
    plt.plot(t, pred, label='prediction')
    plt.xlabel(r'$\tau (fm)$')
    plt.ylabel(r'$d \Gamma / dK$')
    plt.legend()
    plt.show()

from socket import close
import matplotlib
# from pytorch_lightning import Trainer
# from pytorch_lightning import utilities
# from utils.iterators import data_iterators
import numpy as np
import pandas as pd
# from MLP import MLP
# import torch.nn as nn
# import torch

import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')


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
    subset = subset[df['K'] == k_closest]
    return subset


HIDDEN_DIM = 16
DATASET = 'method_0.csv'

P = 16
T = 0.3
K = 3

df = pd.read_csv('datasets/method_0.csv')
df = get_closest_values(df, P, K, T)
print(df)
model_input = df[['M', 'P', 'K', 'T', 't']]
data_gam = df['gamma']
t = df['t']
plt.plot(t, data_gam)
plt.show()

# print(idx)


# # pred = trainer.predict(model, test)
# # pred = torch.vstack(list(chain(*pred)))

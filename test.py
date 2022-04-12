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

AVAIL_GPUS = 0
EPOCHS = 2
BATCH_SIZE = 64
HIDDEN_DIM = 16
SEED = 0
DATASET = 'method_0.csv'
# utilities.seed.seed_everything(seed=SEED)


def visualize_test(P, K, T, model, N=50, df_truth=None):
    # file = 'datasets/combined_rates.csv'
    # df = pd.read_csv(file)

    # df = df['T' == 0.3]
    # print(df.shape)
    # post_results = model(test[0])
    # print(((post_results - init_results)**2).mean().item())

    t = torch.linspace(0, 12, N)

    P = torch.ones(N)*P
    K = torch.ones(N)*K
    T = torch.ones(N)*T

    stack = torch.stack([P, K, T, t]).T
    model.eval()
    out = model(stack)

    plt.plot(t.detach().numpy(), out.detach().numpy())
    plt.show()


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


df = pd.read_csv('datasets/method_0.csv')

P = 16
T = 0.3
K = 3

df = get_closest_values(df, P, K, T)
plt.plot(df['t'], df['gamma'])
plt.show()

# print(idx)


# # pred = trainer.predict(model, test)
# # pred = torch.vstack(list(chain(*pred)))

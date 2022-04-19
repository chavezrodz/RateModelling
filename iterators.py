import os
import torch
import pandas as pd
import torch.utils.data as data
import numpy as np
from utils import get_consts_dict


def data_iterators(datafile, datapath='datasets',
                   batch_size=64, num_workers=8,
                   test_samp=0.1, val_samp=0.2,
                   include_method=False,
                   shuffle_dataset=True
                   ):
    X = pd.read_csv(os.path.join(datapath, datafile)).values

    # Removing negative decay rates
    X = X[np.where(X[:, -1] > 0)]
    # removing method idx
    X = X[:, 1:] if not include_method else X

    # Test splitting for no data leakage in constants
    n_samples = len(X)
    train_samp = 1 - test_samp - val_samp
    train_cutoff = int(train_samp*n_samples)
    val_cutoff = int((train_samp + val_samp)*n_samples)

    consts_dict = get_consts_dict(X[:train_cutoff])

    Q = torch.tensor(X).type(torch.float)
    if shuffle_dataset:
        order = np.arange(n_samples)
        shuffled_order = np.random.shuffle(order)
        Q[order] = Q.clone()[shuffled_order]

    train = data.TensorDataset(Q[:train_cutoff, :-1],
                               Q[:train_cutoff, -1].unsqueeze(1))
    valid = data.TensorDataset(Q[train_cutoff:val_cutoff, :-1],
                               Q[train_cutoff:val_cutoff, -1].unsqueeze(1))
    test = data.TensorDataset(Q[val_cutoff:, :-1],
                              Q[val_cutoff:, -1].unsqueeze(1))

    train_loader = data.DataLoader(train, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
    return (train_loader, valid_loader, test_loader), consts_dict

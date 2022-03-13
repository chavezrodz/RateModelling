import os
import torch
import pandas as pd
import torch.utils.data as data

def data_iterators(datafile='combined_rates.csv', datapath='datasets',
    train_samp=0.8, batch_size=64, num_workers=8):
    X = pd.read_csv(os.path.join(datapath, datafile))
    Q = torch.tensor(X.values).type(torch.float)

    n_samples = Q.shape[0]
    train_cutoff = int(train_samp*n_samples)

    train = data.TensorDataset(Q[:train_cutoff, :3], Q[:train_cutoff, 4].unsqueeze(1))
    valid = data.TensorDataset(Q[train_cutoff:, :3], Q[train_cutoff:, 4].unsqueeze(1))

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_loader, valid_loader

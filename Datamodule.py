import os
import torch
import pandas as pd
import torch.utils.data as data
import numpy as np
from misc.utils import load_df
from misc.norms import get_consts_dict_modelling, get_consts_dict_integrating
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.method = args.method
        self.dataset = args.proj_dir
        self.datapath = args.data_dir
        self.results_path = args.results_dir
        self.which_spacing = args.which_spacing
        self.batch_size = args.batch_size
        self.shuffle_dataset = args.shuffle_dataset
        self.num_workers = args.n_workers
        self.val_samp = args.val_sample
        self.test_samp = 0.1
        self.include_method = False

        datafile = 'method_'+str(self.method)+'.csv'
        print(self.dataset)
        if self.dataset == 'rate_modelling':
            df = load_df(self.datapath, datafile, self.which_spacing)
        elif self.dataset == 'rate_integrating':
            df = pd.read_csv(
                os.path.join(self.results_path, 'integral_results', datafile)
                )
            df.drop_duplicates()
        else:
            raise Exception('Dataset not found')

        X = df.values
        # removing method idx
        X = X[:, 1:] if not self.include_method else X

        # Test splitting for no data leakage in constants
        n_samples = len(X)
        train_samp = 1 - self.test_samp - self.val_samp
        train_cutoff = int(train_samp*n_samples)

        Q = torch.tensor(X).type(torch.float)

        if self.dataset == 'rate_modelling':
            self.consts_dict = get_consts_dict_modelling(Q[:train_cutoff])
        elif self.dataset == 'rate_integrating':
            self.consts_dict = get_consts_dict_integrating(Q[:train_cutoff])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        datafile = 'method_'+str(self.method)+'.csv'
        if self.dataset == 'rate_modelling':
            df = load_df(self.datapath, datafile, self.which_spacing)
        elif self.dataset == 'rate_integrating':
            df = pd.read_csv(
                os.path.join(self.results_path, 'integral_results', datafile)
                )
            df.drop_duplicates()
        else:
            raise Exception('Dataset not found')

        X = df.values
        # removing method idx
        X = X[:, 1:] if not self.include_method else X

        # Test splitting for no data leakage in constants
        n_samples = len(X)
        train_samp = 1 - self.test_samp - self.val_samp
        train_cutoff = int(train_samp*n_samples)
        val_cutoff = int((train_samp + self.val_samp)*n_samples)

        Q = torch.tensor(X).type(torch.float)

        if self.dataset == 'rate_modelling':
            self.consts_dict = get_consts_dict_modelling(Q[:train_cutoff])
        elif self.dataset == 'rate_integrating':
            self.consts_dict = get_consts_dict_integrating(Q[:train_cutoff])

        if self.shuffle_dataset:
            order = np.arange(n_samples)
            shuffled_order = np.random.shuffle(order)
            Q[order] = Q.clone()[shuffled_order]

        if stage in (None, "fit"):
            train_ds = data.TensorDataset(
                Q[:train_cutoff, :-1], Q[:train_cutoff, -1].unsqueeze(1)
                )
            val_ds = data.TensorDataset(
                Q[train_cutoff:val_cutoff, :-1],
                Q[train_cutoff:val_cutoff, -1].unsqueeze(1)
                )
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage in (None, "test"):
            self.test_ds = data.TensorDataset(
                Q[val_cutoff:, :-1], Q[val_cutoff:, -1].unsqueeze(1)
                )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
            )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
            )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
            )

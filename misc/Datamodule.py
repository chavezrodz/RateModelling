import os
import torch
import pandas as pd
import torch.utils.data as data
import numpy as np
from misc.load_df import load_df
from misc.norms import get_consts_dict_modelling, get_consts_dict_integrating
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 method,
                 proj_dir,
                 datapath,
                 results_dir,
                 which_spacing,
                 batch_size=1024,
                 shuffle_dataset=True,
                 num_workers=8,
                 val_samp=0.1,
                 test_samp=0.1,
                 include_method=False,
                 seed=0
                 ):

        super().__init__()
        self.method = method
        self.proj_dir = proj_dir
        self.datapath = datapath
        self.results_path = results_dir
        self.which_spacing = which_spacing

        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.num_workers = num_workers
        self.val_samp = val_samp
        self.test_samp = test_samp
        self.include_method = include_method
        self.seed = seed

        self.input_dim = 3 if proj_dir == 'rate_integrating' else 4

        df = load_df(datapath, proj_dir, method, which_spacing)
        t = np.sort(df['t'].unique())[-1]
        df_pkT = df[df['t'] == t].copy().drop(['t', 'gamma'], axis=1)
        df_pkT = df_pkT.sample(frac=1, random_state=0)
        # Test splitting for no data leakage in constants
        n_samples = len(df_pkT)
        train_samp = 1 - self.test_samp - self.val_samp
        tr_cutoff = int(train_samp*n_samples)

        train_pkT = df_pkT.iloc[:tr_cutoff]
        train_full = train_pkT.merge(df, how='inner', on=['M', 'P', 'K', 'T'])

        Q = torch.tensor(train_full.values).type(torch.float)

        match self.proj_dir:
            case 'rate_modelling':
                self.consts_dict = get_consts_dict_modelling(Q)
            case 'rate_integrating':
                self.consts_dict = get_consts_dict_integrating(Q)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = load_df(
            self.datapath, self.proj_dir, self.method, self.which_spacing)
        t = np.sort(df['t'].unique())[-1]
        df_pkT = df[df['t'] == t].copy().drop(['t', 'gamma'], axis=1)
        df_pkT = df_pkT.sample(frac=1, random_state=0)

        # Test splitting for no data leakage in constants
        n_samples = len(df_pkT)
        train_samp = 1 - self.test_samp - self.val_samp
        tr_cutoff = int(train_samp*n_samples)
        val_cutoff = int((train_samp + self.val_samp)*n_samples)

        train_pkT = df_pkT.iloc[:tr_cutoff].copy()
        val_pkT = df_pkT.iloc[tr_cutoff:val_cutoff].copy()
        test_pkT = df_pkT.iloc[val_cutoff:].copy()

        train_full = train_pkT.merge(df, how='inner', on=['M', 'P', 'K', 'T'])
        train_full = train_full.sample(frac=1, random_state=self.seed)
        val_full = val_pkT.merge(df, how='inner', on=['M', 'P', 'K', 'T'])
        test_full = test_pkT.merge(df, how='inner', on=['M', 'P', 'K', 'T'])
        if not self.include_method:
            train_full = train_full.drop('M', axis=1)
            val_full = val_full.drop('M', axis=1)
            test_full = test_full.drop('M', axis=1)

        Q_tr = torch.tensor(train_full.values).type(torch.float)
        Q_val = torch.tensor(val_full.values).type(torch.float)
        Q_test = torch.tensor(test_full.values).type(torch.float)

        match self.proj_dir:
            case 'rate_modelling':
                self.consts_dict = get_consts_dict_modelling(Q_tr)
            case 'rate_integrating':
                self.consts_dict = get_consts_dict_integrating(Q_tr)

        if stage in (None, "fit"):
            train_ds = data.TensorDataset(
                Q_tr[:, :-1],
                Q_tr[:, -1].unsqueeze(1)
                )
            val_ds = data.TensorDataset(
                Q_val[:, :-1],
                Q_val[:, -1].unsqueeze(1)
                )
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage in (None, "test"):
            self.test_ds = data.TensorDataset(
                Q_test[:, :-1],
                Q_test[:, -1].unsqueeze(1)
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

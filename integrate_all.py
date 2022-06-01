from numbers import Integral
from multiprocessing import Pool
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import os
import itertools
from misc.utils import load_df
from misc.iterators import data_iterators
from misc.MLP import MLP
import scipy.integrate as integrate
import csv


def get_p_Tt_combs(df):
    p_values = np.sort(df['P'].unique())
    T_values = np.sort(df['T'].unique())
    t_values = np.sort(df['t'].unique())
    return p_values, T_values, t_values


def k_integral(P, T, t, model, a, b, N):
    # Model integral
    def func(k):
        k = torch.tensor(k)
        X = torch.zeros(len(k), 4)
        X[:, 0] = P
        X[:, 1] = k
        X[:, 2] = T
        X[:, 3] = t
        return np.array(model(X).detach().squeeze())

    return integrate.quadrature(func, a*P, b*P)[0]


def load_model(args):
    h_dim = args.hidden_dim
    n_layers = args.n_layers
    method = args.method
    results_dir = args.results_dir
    datapath = args.datapath
    batch_size = 64

    pc_err = args.pc_err
    datafile = 'method_'+str(method)+'.csv'

    (_, _, _), consts_dict = data_iterators(
        datafile=datafile,
        datapath=datapath,
        batch_size=batch_size
        )

    model_file = f'M_{method}_n_layers_{n_layers}_hid_dim_{h_dim}'
    model_file += f'_val_pc_err={pc_err}.ckpt'
    model_path = os.path.join(
        results_dir, "Rate_modelling", "saved_models",
        f'Method_{method}', model_file
        )
    model = MLP.load_from_checkpoint(
        checkpoint_path=model_path,
        hidden_dim=h_dim,
        n_layers=n_layers,
        consts_dict=consts_dict
        )
    return model


def main(args):
    datafile = 'method_'+str(args.method)+'.csv'
    datapath = args.datapath
    results_dir = args.results_dir

    a = args.a
    b = args.b
    N = args.N
    n_threads = args.n_threads

    model = load_model(args)
    df = load_df(datapath, datafile, args.which_spacing)
    p_values, T_values, t_values = get_p_Tt_combs(df)
    results_path = os.path.join(
        results_dir,
        'integral_results_single',
        'method_'+str(args.method)
        )
    os.makedirs(results_path, exist_ok=True)
    fieldnames = ['M', 'P', 'T', 't', 'Integral']

    for p in p_values[:1]:
        file_name = f'p_{p}'.replace('.', '_')+'.csv'
        file_name = os.path.join(results_path, file_name)
        with open(file_name, 'w', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for (T, t) in itertools.product(T_values, t_values):
                print(f"Running p:{p} T:{T} t:{t}")
                result = k_integral(p, T, t, model, a, b, N=N)
                row = {
                    'M': int(args.method),
                    'P': p,
                    'T': T,
                    't': t,
                    'Integral': result
                    }
                writer.writerow(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Model Params
    parser.add_argument("--method", default=0, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--pc_err", default='2.93e-02', type=str)

    # Integral Params
    parser.add_argument("--a", default=0.05, type=float)
    parser.add_argument("--b", default=0.95, type=float)
    parser.add_argument("--N", default=50, type=int)

    # Managing params
    parser.add_argument("--n_threads", default=8, type=int)
    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='../datasets', type=str)
    parser.add_argument("--which_spacing", default='both', type=str)
    args = parser.parse_args()

    main(args)

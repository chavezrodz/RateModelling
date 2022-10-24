from multiprocessing import Pool
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import os
import itertools
from misc.utils import load_df
from misc.load_model import load_model
import scipy.integrate as integrate
from Datamodule import DataModule
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


def main(args):
    datafile = 'method_'+str(args.method)+'.csv'
    datapath = args.data_dir
    results_dir = args.results_dir

    a = args.a
    b = args.b
    N = args.N

    dm = DataModule(args)
    model = load_model(args, dm.consts_dict, saved=True)

    df = load_df(datapath, datafile, args.which_spacing)
    p_values, T_values, t_values = get_p_Tt_combs(df)
    single_results_dir = os.path.join(
        results_dir,
        'integral_results_single',
        'method_'+str(args.method)
        )
    os.makedirs(single_results_dir, exist_ok=True)
    fieldnames = ['M', 'P', 'T', 't', 'Integral']

    if args.compute:
        for p in p_values:
            file_name = f'p_{p}'.replace('.', '_')+'.csv'
            file_name = os.path.join(single_results_dir, file_name)
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

    if args.combine:
        dfs = []
        for file in os.listdir(single_results_dir):
            dfs.append(pd.read_csv(os.path.join(single_results_dir, file)))
        df = pd.concat(dfs, ignore_index=True)
        combined_dir = os.path.join(results_dir, 'integral_results')
        os.makedirs(combined_dir, exist_ok=True)

        df.to_csv(
            os.path.join(combined_dir, 'method_'+str(args.method)+'.csv'),
            index=False
            )


if __name__ == '__main__':
    parser = ArgumentParser()

    # Model Params
    parser.add_argument("--method", default=2, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--pc_err", default='2.50e-02', type=str)

    parser.add_argument("--compute", default=True, type=bool)
    parser.add_argument("--combine", default=False, type=bool)

    # Integral Params
    parser.add_argument("--a", default=0.01, type=float)
    parser.add_argument("--b", default=0.99, type=float)
    parser.add_argument("--N", default=50, type=int)

    # Managing params
    parser.add_argument("--n_threads", default=8, type=int)
    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--proj_dir", default='rate_modelling', type=str)
    parser.add_argument("--data_dir", default='../datasets', type=str)
    parser.add_argument("--which_spacing", default='both', type=str)
    args = parser.parse_args()

    main(args)

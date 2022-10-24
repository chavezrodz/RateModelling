from pytorch_lightning import Trainer
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from misc.utils import load_df
import scipy.integrate as integrate
from misc.utils import generate_test_data, get_closest_values, load_df
from misc.load_model import load_model
from Datamodule import DataModule


def compare_with_data(P, K, T, datapath, dataset, model, include_data):
    K *= P
    if include_data:
        df = load_df(datapath, dataset, 'both')
        df = get_closest_values(df, P, K, T)
        df = df.sort_values(by='t', ascending=True)

        data_gam = df['gamma']
        t = df['t'].values
        X = df[['P', 'K', 'T', 't']].values
        P = df['P'].iloc[0]
        K = df['K'].iloc[0]
        T = df['T'].iloc[0]

        X = torch.tensor(X).type(torch.float)
        plt.plot(t, data_gam, label='data')

    else:
        X = generate_test_data(P, K, T)
        t = X[:, -1].clone()
        err = 0

    pred = model(X).detach().squeeze()

    if include_data:
        err = ((pred-data_gam)/data_gam).abs().mean()

    print(f"""
        P = {P}
        K = {K}
        T = {T}
    Error: {err*100:.2f}%
    """)

    plt.plot(t, pred, label='prediction')
    plt.xlabel(r'$\tau (fm)$')
    plt.ylabel(r'$d \Gamma / dK$')
    plt.legend()
    plt.show()


def compare_k_preds(P, T, t, datapath, dataset, model,
                    a, b, N=50, which='both'):
    # Data
    df = load_df(datapath, dataset, which)

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
    print(subset['K'].min(), subset['K'].max())
    plt.plot(subset['K'], subset['gamma'], label='data')

    # Model
    K = torch.linspace(a, b, N) * P
    X = torch.zeros(N, 4)
    X[:, 0] = P
    X[:, 1] = K
    X[:, 2] = T
    X[:, 3] = t
    pred = model(X).detach().squeeze()

    plt.plot(K, pred, label='prediction')
    plt.legend()
    plt.show()

    pass


def k_integral(P, T, t, datapath, dataset, model, a, b, N=50, which='both'):
    # Data
    df = load_df(datapath, dataset, which)

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
    simp_integral = integrate.simpson(y, x=x)
    # Model

    def func(k):
        k = torch.tensor(k)
        X = torch.zeros(len(k), 4)
        X[:, 0] = P
        X[:, 1] = k
        X[:, 2] = T
        X[:, 3] = t
        return np.array(model(X).detach().squeeze())

    model_integral = integrate.quadrature(func, a*P, b*P)

    print(f"""
    Simpson_integral:{simp_integral}
    Model Integral with Gaussian Quadrature: {model_integral}
    """)

    return


def main(args):
    datafile = 'method_'+str(args.method)+'.csv'
    datapath = args.data_dir
    results_dir = args.results_dir

    dm = DataModule(args, batch_size=8)
    model = load_model(args, dm, saved=True)

    df = load_df(datapath, datafile, args.which_spacing)

    if args.test:
        if args.proj_dir == "rate_modelling":
            P = args.P
            K = args.K
            T = args.T
            t = args.t

            a = args.a
            b = args.b
            N = args.N
            which = args.which_spacing
            compare_with_data(P, K, T, datapath, datafile, model,
                              include_data=args.include_data)
            compare_k_preds(P, T, t, datapath, datafile, model, a, b, N=N,
                            which=which)
            k_integral(P, T, t, datapath, datafile, model,
                       a, b, N=N, which=which)

        elif args.proj_dir == "rate_integrating":
            pass

    if args.export:
        dm.setup(stage='test')
        for batch in dm.test_dataloader():
            x = batch[0]
            input_example = x[:4]
            break
            # print(input_example)
            # print(model(input_example))

        compiled_dir = os.path.join(
            results_dir,
            "compiled_models",
            args.proj_dir,
            )
        os.makedirs(compiled_dir, exist_ok=True)

        model.to_torchscript(
            file_path=os.path.join(compiled_dir, f'Method_{args.method}.pt'),
            example_inputs=input_example
            )


if __name__ == '__main__':
    parser = ArgumentParser()
    # Managing params
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--export", default=True, type=bool)

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--proj_dir", default='rate_integrating', type=str)
    parser.add_argument("--data_dir", default='../datasets', type=str)
    parser.add_argument("--which_spacing", default='both', type=str)

    # Model Params
    parser.add_argument("--method", default=3, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--pc_err", default='4.90e-02', type=str)

    # Rate Modelling
    parser.add_argument("--t", default=1.05, type=float)
    parser.add_argument("--a", default=0.01, type=float)
    parser.add_argument("--b", default=0.99, type=float)
    parser.add_argument("--N", default=50, type=int)

    # Rate Modelling & Integrating
    parser.add_argument("--include_data", default=True, type=bool)
    parser.add_argument("--P", default=100, type=float)
    parser.add_argument("--K", default=0.09, type=float)
    parser.add_argument("--T", default=0.26, type=float)

    # Rate Integrating
    args = parser.parse_args()

    main(args)

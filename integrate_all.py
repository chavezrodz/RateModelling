from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import os
import itertools
from misc.load_df import load_df
import scipy.integrate as integrate
import csv

# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass


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

    return integrate.quadrature(func, a * P, b * P)[0]


def func(args):
    model = args[0]
    p = args[1]
    outdir = args[2]
    T_values = args[3]
    t_values = args[4]
    a, b, N = args[5]

    file_name = f"p_{p}".replace(".", "_") + ".csv"
    file_name = os.path.join(outdir, file_name)
    with open(file_name, "w", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["P", "T", "t", "Integral"])
        writer.writeheader()
        for T, t in itertools.product(T_values, t_values):
            result = k_integral(p, T, t, model, a, b, N=N)
            row = {"P": p, "T": T, "t": t, "Integral": result}
            writer.writerow(row)


def main(args):
    for method in range(4):
        single_results_dir = os.path.join(
            args.results_dir,
            "rate_integrating",
            "Int_results",
            "divided",
            "method_" + str(method),
        )
        if args.compute:
            model = torch.jit.load(
                os.path.join(
                    "Results",
                    "compiled_models",
                    "rate_modelling",
                    f"Method_{method}.pt",
                )
            )

            os.makedirs(single_results_dir, exist_ok=True)
            df = load_df(args.data_dir, "rate_modelling", method, args.which_spacing)
            p_values = np.sort(df["P"].unique())
            T_values = np.sort(df["T"].unique())
            t_values = np.sort(df["t"].unique())

            for p in p_values:
                func(
                    [
                        model,
                        p,
                        single_results_dir,
                        T_values,
                        t_values,
                        (args.a, args.b, args.N),
                    ]
                )

            # Cant use MP while using a compiled torch model, an attempt
            # clones, outfiles, T_sets, t_sets, int_pars = [], [], [], [], []
            # for i in range(len(p_values)):
            # clones.append(deepcopy(model))
            # outfiles.append(single_results_dir)
            # T_sets.append(T_values)
            # t_sets.append(t_values)
            # int_pars.append((args.a, args.b, args.N))

            # zipped_args = list(zip(
            #     clones, p_values, outfiles, T_sets, t_sets, int_pars
            #     ))

            # mypool = Pool(args.n_threads)
            # mypool.map(func, zipped_args)

        if args.combine:
            combined_dir = os.path.join(
                args.results_dir, "rate_integrating", "int_results", "combined"
            )
            os.makedirs(combined_dir, exist_ok=True)

            dfs = []
            for file in os.listdir(single_results_dir):
                dfs.append(pd.read_csv(os.path.join(single_results_dir, file)))
            df = pd.concat(dfs, ignore_index=True)
            df.insert(loc=0, column="M", value=method)
            df.to_csv(
                os.path.join(combined_dir, "method_" + str(method) + ".csv"),
                index=False,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    # Managing params
    parser.add_argument("--n_threads", default=8, type=int)
    parser.add_argument("--results_dir", default="Results", type=str)
    parser.add_argument("--data_dir", default="../datasets", type=str)
    parser.add_argument("--which_spacing", default="linspaced", type=str)

    # Integral Params
    parser.add_argument("--a", default=0.01, type=float)
    parser.add_argument("--b", default=0.99, type=float)
    parser.add_argument("--N", default=50, type=int)

    parser.add_argument("--compute", default=False, type=bool)
    parser.add_argument("--combine", default=False, type=bool)

    args = parser.parse_args()
    main(args)

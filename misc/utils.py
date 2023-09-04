import numpy as np
import os


def closest_value(df, col, val):
    uniques = df[col].unique()
    close_idx = np.argmin(np.abs(uniques - val))
    return uniques[close_idx]


def get_closest_values(df, P, K, T):
    """
    K enters in units of P
    out as real K
    """
    p_closest = closest_value(df, 'P', P)
    T_closest = closest_value(df, 'T', T)
    subset = df[(df['P'] == p_closest) & (df['T'] == T_closest)]
    k_closest = closest_value(subset, 'K', K*p_closest)
    subset = subset.loc[df['K'] == k_closest]
    return subset['P'].iloc[0], subset['K'].iloc[0], subset['T'].iloc[0]


def make_file_prefix(args):
    file_prefix = 'M_' + str(args.method)
    file_prefix += '_n_layers_' + str(args.n_layers)
    file_prefix += '_hid_dim_' + str(args.hidden_dim)
    file_prefix += '_val_pc_err='
    return file_prefix


def make_checkpt_dir(args):
    dir = os.path.join(
        args.results_dir,
        args.proj_dir,
        "saved_models",
        args.which_spacing,
        'Method_'+str(args.method)
        )
    return dir
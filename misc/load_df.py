import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def preprocess(datapath, method, spacing):
    print(f"Processing {spacing} method {method}")
    datafile = 'method_'+str(method)+'.csv'

    df_file = os.path.join(
        datapath, 'rate_modelling', 'processed', spacing, datafile
        )
    os.makedirs(os.path.join(
        datapath, 'rate_modelling', 'processed', spacing), exist_ok=True)

    if spacing == 'both':
        df1 = pd.read_csv(os.path.join(
            datapath, 'rate_modelling', 'raw', 'logspaced', datafile))
        df2 = pd.read_csv(os.path.join(
            datapath, 'rate_modelling', 'raw', 'linspaced', datafile))
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(
            datapath, 'rate_modelling', 'raw', spacing, datafile))
    # n_samples = 100000
    # df = df.drop_duplicates()[:n_samples]
    df = df.drop_duplicates()

    # Setting metrics on mid pt to chose PKT
    mid_pt = 15
    init_len = len(df.copy())
    ts = np.sort(df['t'].unique())
    t_mid_idx = np.argmin(np.abs(ts - mid_pt))
    t_mid = ts[t_mid_idx]
    df_mid = df[df['t'] == t_mid].sort_values(by=['P', 'K', 'T'], ascending=True)

    df_mid = df_mid.drop(['t'], axis=1)
    df_mid['pc_err'] = np.zeros(len(df_mid))
    df_mid['tot_change'] = np.zeros(len(df_mid))
    df_mid['end_change'] = np.zeros(len(df_mid))
    for index, row in df_mid.iterrows():
        P, K, T, gam = row['P'], row['K'], row['T'], row['gamma']
        idx_subset = df.loc[
            ((df['P'] == P) & (df['K'] == K) & (df['T'] == T)), :]
        subset = df.loc[idx_subset.index.values].sort_values(
            by='t', ascending=True)
        gammas = subset['gamma'].values
        pc_diff = ((gammas[1:] - gammas[:-1])/gammas[1:])*100
        pc_diff = np.where(np.abs(pc_diff)>0.5, pc_diff, 0)

        end_change = np.sign(pc_diff[t_mid_idx:])
        end_change = np.where(end_change < 0, end_change, 0)
        end_change = np.sum(end_change)
        # end_change = np.sum(np.sign(pc_diff)[t_mid_idx:])
        total_change = np.abs(np.diff(np.sign(pc_diff))).sum()
        df_mid.loc[index, 'pc_err'] = np.abs((gam - subset['gamma'].values)/gam).max()*100
        df_mid.loc[index, 'tot_change'] = total_change
        df_mid.loc[index, 'end_change'] = end_change

    # sorting function, to chose PKT
    match method:
        case 0:
            th_pc = 100
            th_chg = 3
            idx_bad_pkT = np.where(
                (df_mid['pc_err'] > th_pc) & (df_mid['tot_change'] > th_chg) | 
                (df_mid['end_change'] < 0) |
                (df_mid['tot_change'] > 5) |
                (df_mid['pc_err'] > 100)
                )[0]
        case 1:
            idx_bad_pkT = np.where(
                (df_mid['tot_change'] > 5) |
                (df_mid['pc_err'] > 105)
                )[0]
        case 2:
            # th_pc = 1
            # th_chg = 150
            idx_bad_pkT = np.where(
                # (df_mid['pc_err'] > th_pc) & (df_mid['tot_change'] > th_chg) | 
                (df_mid['tot_change'] > 5) |
                (df_mid['end_change'] < 0) |
                (df_mid['pc_err'] > 100)
                )[0]
        case 3:
            th_pc = 500
            th_chg = 10
            idx_bad_pkT = np.where(
                (df_mid['pc_err'] > th_pc) & (df_mid['tot_change'] > th_chg)
                )[0]

    df_bad_pkT = df_mid.iloc[idx_bad_pkT].copy().sort_values(
            by='tot_change', ascending=True)

    # Obtain and dislay good values of pkT
    idx_good_pkT = df_mid.index.difference(df_bad_pkT.index)
    df_good_pkT = df_mid.loc[idx_good_pkT].copy()

    fig, axs = plt.subplots(2, 6)
    axs[0, 0].set_ylabel('Before Cleaning')
    axs[1, 0].set_ylabel('After Cleaning')
    axs[1, 0].set_xlabel('pc change')
    axs[1, 1].set_xlabel('oscillation metric')
    axs[1, 2].set_xlabel('end change')
    axs[1, 3].set_xlabel('P')
    axs[1, 4].set_xlabel('K')
    axs[1, 5].set_xlabel('T')
    axs[0, 0].hist(df_mid['pc_err'])
    axs[0, 1].hist(df_mid['tot_change'])
    axs[0, 2].hist(df_mid['end_change'])
    axs[0, 3].hist(df_mid['P'])
    axs[0, 4].hist(df_mid['K'])
    axs[0, 5].hist(df_mid['T'])
    axs[1, 0].hist(df_good_pkT['pc_err'])
    axs[1, 1].hist(df_good_pkT['tot_change'])
    axs[1, 2].hist(df_good_pkT['end_change'])
    axs[1, 3].hist(df_good_pkT['P'])
    axs[1, 4].hist(df_good_pkT['K'])
    axs[1, 5].hist(df_good_pkT['T'])
    axs[1, 3].set_xscale('log')
    axs[0, 3].set_xscale('log')
    axs[1, 4].set_xscale('log')
    axs[0, 4].set_xscale('log')

    plt.tight_layout()
    plt.show()

    print(
        f'Number of corrupt PKT: {len(df_bad_pkT)}/{len(df_mid)} {len(df_bad_pkT)/len(df_mid):%}')

    # n_last = len(df_good_pkT)
    n_last = len(df_mid)//10
    # n_last = 100

    df_good_pc_chg = df_good_pkT.sort_values(
        by='pc_err', ascending=True)[-n_last:].copy()
    df_good_t_chg = df_good_pkT.sort_values(
        by='tot_change', ascending=True)[-n_last:].copy()
    fig, axs = plt.subplots(2, 1)
    oscs, errs = [], []
    for i in range(n_last):
        row = df_good_pc_chg.iloc[-i]
        P, K, T, gam = row['P'], row['K'], row['T'], row['gamma']
        errs.append(row['pc_err'])
        idx = df.loc[((df['P'] == P) & (df['K'] == K) & (df['T'] == T)), :]
        subset = df.loc[idx.index.values].sort_values(by='t', ascending=True)
        # axs[0].plot(subset['t'], subset['gamma']/np.abs(subset['gamma']).max())
        axs[0].plot(subset['t'], subset['gamma']/gam)
        # print(f"Oscillation Value {row['tot_change']} pc chg {row['pc_err']}")

        row = df_good_t_chg.iloc[-i]
        P, K, T, gam = row['P'], row['K'], row['T'], row['gamma']
        oscs.append(row['tot_change'])
        idx = df.loc[((df['P'] == P) & (df['K'] == K) & (df['T'] == T)), :]
        subset = df.loc[idx.index.values].sort_values(by='t', ascending=True)
        axs[1].plot(subset['t'], subset['gamma']/gam)
        # print(f"Oscillation Value {row['tot_change']}")
    axs[0].set_title(f'Worse {n_last} by % error, avg: {np.mean(errs):f}%')
    axs[1].set_title(f'Worse {n_last} by Oscillations, avg: {np.mean(oscs):f}')
    plt.tight_layout()
    plt.show()

    # Show bad values of pkT and getting time indices to remove from df
    # df_bad_pc_chg = df_bad_pkT.sort_values(by='pc_err', ascending=True).copy()
    # df_bad_t_chg = df_bad_pkT.sort_values(by='tot_change', ascending=True).copy()

    fig, axs = plt.subplots(2, 1)
    indices_bad_pkTt = []
    oscs, errs = [], []
    for index in range(len(df_bad_pkT)):
        row = df_bad_pkT.iloc[index]
        P, K, T, gam = row['P'], row['K'], row['T'], row['gamma']
        oscs.append(row['tot_change']), errs.append(row['pc_err'])
        # print(f'P:{P}, K:{K}, T:{T} is above threshold')
        idx_1 = df.loc[((df['P'] == P) & (df['K'] == K) & (df['T'] == T) & (df['t'] < t_mid)), :]
        subset = df.loc[idx_1.index.values].sort_values(by='t', ascending=True)
        axs[0].plot(subset['t'], subset['gamma']/gam)

        idx_2 = df.loc[((df['P'] == P) & (df['K'] == K) & (df['T'] == T) & (df['t'] > t_mid)), :]
        subset = df.loc[idx_2.index.values].sort_values(by='t', ascending=True)
        axs[1].plot(subset['t'], subset['gamma']/np.abs(np.mean(subset['gamma'])))

        idx_full = df.loc[((df['P'] == P) & (df['K'] == K) & (df['T'] == T)), :]
        indices_bad_pkTt.extend(idx_full.index.values)
    fig.suptitle(f'Bad ones before and after cutoff \n avg osc.:{np.mean(oscs):f} avg err:{np.mean(errs):e}')
    plt.show()

    indices_bad_pkTt = pd.Index(indices_bad_pkTt)
    indices_good_pkTt = df.index.difference(indices_bad_pkTt)

    df = df.loc[indices_good_pkTt]
    print(f'Total points used {len(df)}/{init_len} ({len(df)/init_len:%})')
    df.to_csv(df_file, index=False)
    return df


def load_df(datapath, proj_dir, method, spacing='logspaced'):
    datafile = 'method_'+str(method)+'.csv'
    match proj_dir:
        case 'rate_modelling':
            df_file = os.path.join(
                datapath, proj_dir, 'processed', spacing, datafile)
        case 'rate_integrating':
            df_file = os.path.join(
                'Results', 'integral_results', datafile)
    df = pd.read_csv(df_file)
    return df.drop_duplicates()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations as combinations


def make_pairplots(df, savepath):
    cols = df.columns[:-1]
    combs = list(combinations(cols, 2))

    z = df['gamma']

    fig, axs = plt.subplots(2, 3, constrained_layout=True)

    for ax, (x_id, y_id) in zip(axs.ravel(), combs):
        x = df[x_id]
        y = df[y_id]
        im = ax.scatter(x, y, c=z, alpha=0.5, cmap='Greys')
        ax.set_ylabel(y_id)
        ax.set_xlabel(x_id)

    fig.colorbar(im, ax=axs.ravel().tolist(), location='right')
    plt.savefig(savepath)


if __name__ == '__main__':
    file = 'datasets/combined_rates.csv'
    df = pd.read_csv(file)
    make_pairplots(df, 'datasets/pairs.png')

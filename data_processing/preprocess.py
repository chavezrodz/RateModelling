import csv
import os
import sys
import shutil
import numpy as np
import subprocess
import glob
import shutil
import pandas as pd
import seaborn as sns

def read_pkt(fn):
    fn = fn.split("P")[-1]
    P, fn = fn.split("K")
    K, fn = fn.split("T")
    T = float(fn[1:-4].replace("_", "."))
    P, K = float(P.strip("_")), float(K.strip("_"))
    return P, K, T

def get_rows(filelist):
    rows = list()
    for fn in filelist:
        P, K, T = read_pkt(fn)
        ts, gs = np.loadtxt(os.path.join(results_dir, fn)).T
        for i, t in enumerate(ts):
            row = {
            'P': P,
            'K': K,
            'T': T,
            't': t,
            'gamma': gs[i]}
            rows.append(row)
    return rows

results_dir = "datasets/Results"
filelist = os.listdir(results_dir)
file_name = 'combined_rates.csv'
file_name = os.path.join('datasets', file_name)

fieldnames = ['P', 'K', 'T', 't', 'gamma']


def preprocess():
    rows = get_rows(filelist)
    if os.path.isfile(file_name):
        with open(file_name, 'a', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for row in rows:
              writer.writerow(row)

    else:
      with open(file_name, 'w', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
          writer.writerow(row)

df = pd.read_csv(file_name)
df = df.drop(["T"],axis=1)


# x = df['P']
# y = df['K']
# z = df['gamma']

# c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z.min(), vmax=z.max())
# ax.set_title('pcolormesh')
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(c, ax=ax)

sns.heatmap(df.pivot("P", "K", "gamma"))
plt.show()

# plt.scatter(df.)
# g = sns.pairplot(df, hue='gamma', diag_kind="kde", corner=True)
# g.map_lower(sns.kdeplot, levels=4, color=".2")
# plt.show()

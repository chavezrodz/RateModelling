import csv
import os
import sys
import shutil
import numpy as np
import subprocess
import glob
import shutil

dest_dir = "schro_results"
os.makedirs(dest_dir, exist_ok=True)

file_lists = [r'schro/r100*', r'schro/sp*']

for file_list in file_lists:
    for file in glob.glob(file_list):
        shutil.move(file, dest_dir)

arr = np.loadtxt(os.path.join(dest_dir, 'r100'))

T = 100
p = 70
k = 30
t = arr[:, 0]
r = arr[:, 1]

with open(os.path.join(dest_dir, 'tabular_results.csv'),
          'w', newline='') as csvfile:
    fieldnames = ['E','p', 'k', 't', 'r']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(t)):
        writer.writerow(
            {'E': T, 'p': p, 'k': k, 't': t[i], 'r': r[i]}
            )

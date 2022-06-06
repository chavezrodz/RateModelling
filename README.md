# How to run

## Requires

python version 3.9.0
requirements.txt

datasets-/linspaced/method_*.csv
        -/logspaced/method_*.csv

## Steps

### 1 generate datasets with schrodinger

### 2 train first MLP with rate data

python train.py --proj_dir rate_modelling --method method_idx

### 3 use first MLP to compute analytical integrals with the model

python integrate_all.py

### 4 train second MLP using analytical rates

python train.py --proj_dir rate_integrating --method method_idx

### Export all models to cpp

python export_all.py

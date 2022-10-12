# How to run

## Requires

python version 3.9.0
requirements.txt

datasets-/linspaced/method_*.csv
        -/logspaced/method_*.csv

## Steps

### 1 Generate datasets with schrodinger code

### 2 Train first MLP with rate data

python train.py --proj_dir rate_modelling --method method_idx

### 3 Use first MLP to compute analytical integrals with the model

python integrate_all.py

### 4 Train second MLP using analytical rates

python train.py --proj_dir rate_integrating --method method_idx

### 5 Export all models to cpp

python test_export.py --proj_dir rate_modelling --method method_idx
python test_export.py --proj_dir rate_integrating --method method_idx

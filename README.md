# How to run

These are the scripts to train and test MLPs for diferential and total decay rates to be used in MARTINI, from the calculations from CH-G

```latex
@article{Caron-Huot:2010qjx,
    author = "Caron-Huot, Simon and Gale, Charles",
    title = "{Finite-size effects on the radiative energy loss of a fast parton in hot and dense strongly interacting matter}",
    eprint = "1006.2379",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevC.82.064902",
    journal = "Phys. Rev. C",
    volume = "82",
    pages = "064902",
    year = "2010"
}
```


## Requires

python version 3.11.6

requirements.txt

datasets-/linspaced/method_*.csv
        -/logspaced/method_*.csv

## Steps

### 1 Generate datasets with schrodinger code

### 2 Train first MLP with rate data

python train.py --proj_dir rate_modelling --method method_idx

### Compile first MLP batch

python export_models.py --proj_dir rate_modelling

### 3 Use first MLP to compute analytical integrals with the model

python integrate_all.py

### 4 Train second MLP using analytical rates

python train.py --proj_dir rate_integrating --method method_idx

### 5 Compile second batch of MLPs

python export_models.py --proj_dir rate_integrating

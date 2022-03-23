from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from utils.iterators import data_iterators
import pandas as pd
from MLP import MLP
import torch.nn as nn
import torch

# from itertools import chain

AVAIL_GPUS = 0
EPOCHS = 2
BATCH_SIZE = 16
HIDDEN_DIM = 16
SEED = 0

utilities.seed.seed_everything(seed=SEED)

# Need to add test set as single batch
train, val, test = data_iterators(batch_size=BATCH_SIZE)

model = MLP(hidden_dim=HIDDEN_DIM)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS
)

trainer.fit(model, train, val)

# pred = trainer.predict(model, test)
# pred = torch.vstack(list(chain(*pred)))

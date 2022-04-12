from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.iterators import data_iterators
import pandas as pd
from MLP import MLP
import torch.nn as nn
import torch


AVAIL_GPUS = 0
EPOCHS = 3
BATCH_SIZE = 512
HIDDEN_DIM = 16
SEED = 0
DATASET = 'method_0.csv'

mode = 'traifdasn'

utilities.seed.seed_everything(seed=SEED)

train, val, test = data_iterators(batch_size=BATCH_SIZE,
                                  datafile=DATASET)


checkpoint_callback = ModelCheckpoint(
    # dirpath="my/path/",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="checkpoint_M0-{epoch:02d}-{val_loss:.3f}",
    save_last=True
    )

if mode == 'train':
    model = MLP(hidden_dim=HIDDEN_DIM)
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback]
        )

    trainer.fit(model, train, val)
    checkpoint_callback.best_model_path
    print(checkpoint_callback.best_model_path)

else:
    model = MLP.load_from_checkpoint(
        checkpoint_path='lightning_logs/version_9/checkpoints/last.ckpt',
        hidden_dim=HIDDEN_DIM
        )

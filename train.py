from re import M
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.iterators import data_iterators
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from MLP import MLP
import torch.nn as nn
import torch


AVAIL_GPUS = 0
EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_DIM = 32
SEED = 0
METHOD = 0
DATASET = 'method_'+str(METHOD)+'.csv'

mode = 'train'

utilities.seed.seed_everything(seed=SEED)

train, val, test = data_iterators(
    batch_size=BATCH_SIZE,
    datafile=DATASET
    )


checkpoint_callback = ModelCheckpoint(
    # dirpath="my/path/",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename=f"Method_{METHOD}_"+"{epoch:02d}-{val_loss:.3f}",
    save_last=False
    )

if mode == 'train':
    model = MLP(hidden_dim=HIDDEN_DIM)
    wandb_logger = WandbLogger(
        project="schrodinger_rates",
        offline=True
        )

    trainer = Trainer(
        logger=wandb_logger,
        gpus=AVAIL_GPUS,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50
        )

    trainer.fit(model, train, val)
    # checkpoint_callback.best_model_path
    # print(checkpoint_callback.best_model_path)

# else:
#     model = MLP.load_from_checkpoint(
#         checkpoint_path='lightning_logs/version_9/checkpoints/last.ckpt',
#         hidden_dim=HIDDEN_DIM
#         )

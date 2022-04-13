from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.callbacks import ModelCheckpoint
from iterators import data_iterators
from pytorch_lightning.loggers import TensorBoardLogger
from MLP import MLP

AVAIL_GPUS = 0

hpars = dict(
    BATCH_SIZE=4096,
    HIDDEN_DIM=128,
    METHOD=0,
)

tpars = dict(
    SEED=0,
    criterion='l1_loss',
    lr=1e-2,
    EPOCHS=10,
    shuffle_dataset=True,
)


datafile = 'method_'+str(hpars['METHOD'])+'.csv'
utilities.seed.seed_everything(seed=tpars['SEED'])

train, val, test = data_iterators(
    batch_size=hpars['BATCH_SIZE'],
    datafile=datafile,
    shuffle_dataset=tpars['shuffle_dataset']
    )


checkpoint_callback = ModelCheckpoint(
    dirpath="saved_models",
    save_top_k=5,
    monitor="valid/"+tpars['criterion'],
    mode="min",
    filename=f"M_{hpars['METHOD']}_"+"{epoch:02d}",
    save_last=False
    )

model = MLP(
    hidden_dim=hpars['HIDDEN_DIM'],
    criterion=tpars['criterion'],
    lr=tpars['lr'],
    )

logger = TensorBoardLogger(
    save_dir='TB_logs',
    default_hp_metric=True
)


trainer = Trainer(
    logger=logger,
    gpus=AVAIL_GPUS,
    max_epochs=tpars['EPOCHS'],
    callbacks=[checkpoint_callback],
    log_every_n_steps=10
    )

trainer.fit(
    model,
    train,
    val
    )

allpars = hpars | tpars
logger.log_hyperparams(
    allpars
    )

# if mode == 'train':
# else:
#     model = MLP.load_from_checkpoint(
#         checkpoint_path='lightning_logs/version_9/checkpoints/last.ckpt',
#         hidden_dim=HIDDEN_DIM
#         )

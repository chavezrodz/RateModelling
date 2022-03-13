from pytorch_lightning import Trainer
from data_processing.iterators import data_iterators
from model import MLP

AVAIL_GPUS = 0
EPOCHS = 10
BATCH_SIZE = 16
HIDDEN_DIM = 16

train, val = data_iterators(batch_size=BATCH_SIZE)
model = MLP(hidden_dim=HIDDEN_DIM)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS
)

trainer.fit(model, train)

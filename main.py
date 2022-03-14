from pytorch_lightning import Trainer
from data_processing.iterators import data_iterators
import pandas as pd
from model import MLP
import torch.nn as nn
import torch
from itertools import chain
import matplotlib.pyplot as plt


def visualize_test(P, K, T, model):
    t = torch.tensor([0.05,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,8.0,12.0])
    P = torch.ones(len(t))*P
    K = torch.ones(len(t))*K
    T = torch.ones(len(t))*T

    stack = torch.stack([P, K, T, t]).T
    out = model(stack)

    plt.plot(t.detach().numpy(), out.detach().numpy())
    plt.show()


AVAIL_GPUS = 0
EPOCHS = 50
BATCH_SIZE = 16
HIDDEN_DIM = 16


# Need to add test set as single batch
train, val, test = data_iterators(batch_size=BATCH_SIZE)

model = MLP(hidden_dim=HIDDEN_DIM)

# init_results = model(test[0])
headers = ['P [GeV]', 'K [GeV]', 'T [GeV]', 't', 'gamma']

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS
)

trainer.fit(model, train, val)
# pred = trainer.predict(model, test)
# pred = torch.vstack(list(chain(*pred)))


# file = 'datasets/combined_rates.csv'
# df = pd.read_csv(file)

# df = df['T' == 0.3]
# print(df.shape)


# post_results = model(test[0])
# print(((post_results - init_results)**2).mean().item())

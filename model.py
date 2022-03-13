import os
import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule

class MLP(LightningModule):
    def __init__(self, hidden_dim, input_dim=3, output_dim=1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        return self.mlp(x)

    def step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_func(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
      return self.step(batch, batch_idx)

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=1e-3)

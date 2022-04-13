import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule


def pc_err(pred, y):
    return ((pred-y).abs()/y).mean()


class MLP(LightningModule):
    def __init__(
        self,
        hidden_dim,
        input_dim=4,
        output_dim=1,
        lr=1e-3,
        criterion='l1_loss',
    ):
        super().__init__()

        self.pc_err = pc_err
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.criterion = criterion
        self.lr = lr
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

    def get_metrics(self, pred, y):
        metrics = dict(
            l1_loss=self.l1_loss(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
        )
        return metrics

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'train/{k}': v for k, v in metrics.items()},
            on_epoch=True
            )
        return metrics[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'valid/{k}': v for k, v in metrics.items()},
            on_epoch=True
            )

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

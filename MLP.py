import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from utils import normalize_vector, pc_err


class MLP(LightningModule):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        input_dim=4,
        output_dim=1,
        lr=1e-3,
        amsgrad=True,
        criterion='abs_err',
    ):
        super().__init__()

        self.pc_err = pc_err
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad

        self.p_m, self.p_n = 1.8494850881985143, 1.1505149978319906
        self.t_m, self.t_n = 0.013138785488880289, 1.288377063031596
        self.T_m, self.T_n = 0.525, 0.475

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            )
        for i in range(n_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = normalize_vector(
            x,
            self.p_m, self.p_n,
            self.T_m, self.T_n,
            self.t_m, self.t_n
            )
        return self.mlp(x).square()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
        )
        return metrics

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'train/{k}': v for k, v in metrics.items()},
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
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad
            )

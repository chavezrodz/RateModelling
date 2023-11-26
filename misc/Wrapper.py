import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule


def MAPE(y_pred, y):
    return ((y - y_pred)/y).abs().mean()


class Wrapper(LightningModule):
    def __init__(
        self,
        core_model,
        consts_dict,
        normalize_func,
        lr=1e-3,
        amsgrad=True,
        criterion='pc_err'
    ):
        super().__init__()
        self.core_model = core_model
        self.consts_dict = consts_dict

        self.pc_err = MAPE
        self.mse = nn.MSELoss()
        self.abs_err = nn.L1Loss()

        self.criterion = criterion
        self.learning_rate = lr
        self.amsgrad = amsgrad
        self.normalize_func = normalize_func

    def forward(self, x):
        x = self.normalize_func(x, self.consts_dict)
        return self.core_model(x).square()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

    def get_metrics(self, pred, y):
        # Always learning the exponents
        metrics = dict(
            pc_err=self.pc_err(pred, y),
            log_mse=self.mse(torch.log1p(pred), torch.log1p(y)),
            log_pc_err=self.pc_err(torch.log1p(pred), torch.log1p(y)),
            log_abs_err=self.abs_err(torch.log1p(pred), torch.log1p(y)),
            )
        return metrics

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'{k}/train': v for k, v in metrics.items()},
            on_epoch=True, on_step=False
            )
        return metrics[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'{k}/validation': v for k, v in metrics.items()},
            on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'{k}': v for k, v in metrics.items()},
            on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            amsgrad=self.amsgrad
            )
        return optimizer

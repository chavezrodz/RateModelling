from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from misc.Datamodule import DataModule
from misc.utils import make_file_prefix, make_checkpt_dir
from misc.load_model import load_model
from torch import manual_seed
import numpy as np
import os


def main(args):
    manual_seed(seed=args.seed)
    np.random.seed(args.seed)

    checkpoint_callback = ModelCheckpoint(
        dirpath=make_checkpt_dir(args),
        save_top_k=1,
        monitor=args.criterion+'/validation',
        mode="min",
        filename=make_file_prefix(args)+'{pc_err/validation:.2e}',
        auto_insert_metric_name=False,
        save_last=False
        )

    dm = DataModule(
        args.method,
        args.proj_dir,
        args.data_dir,
        args.results_dir,
        args.which_spacing,
        batch_size=args.batch_size,
        shuffle_dataset=args.shuffle_dataset,
        num_workers=8,
        val_samp=args.val_samp,
        )

    model = load_model(args.method,
                       args.hidden_dim,
                       args.n_layers,
                       args.proj_dir,
                       dm,
                       saved=False,
                       results_dir=args.results_dir,
                       amsgrad=args.amsgrad,
                       criterion=args.criterion,
                       lr=args.lr
                       )

    logger = TensorBoardLogger(
        save_dir=os.path.join(
            args.results_dir,
            args.proj_dir,
            "TB_logs"
            ),
        name='Method_'+str(args.method),
        default_hp_metric=True)
    logger.log_hyperparams(args)

    trainer = Trainer(
        logger=logger,
        accelerator='auto',
        devices='auto',
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--method", default=3, type=int)

    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='pc_err', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--which_spacing", default='both', type=str)
    parser.add_argument("--proj_dir", default='rate_modelling', type=str,
                        choices=['rate_modelling', 'rate_integrating'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--data_dir", default='../datasets', type=str)
    parser.add_argument("--shuffle_dataset", default=True, type=bool)
    parser.add_argument("--val_samp", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fast_dev_run", default=True, type=bool)
    args = parser.parse_args()

    main(args)

from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from misc.iterators import data_iterators
from misc.utils import make_file_prefix, make_checkpt_dir
from misc.MLP import MLP
import os


def main(args):
    if args.gpu:
        avail_gpus = 1
        n_workers = 0
    else:
        avail_gpus = 0
        n_workers = 8

    utilities.seed.seed_everything(seed=args.seed, workers=True)

    datafile = 'method_'+str(args.method)+'.csv'

    (train_dl, val_dl, test_dl), consts_dict = data_iterators(
        which_spacing=args.which_spacing,
        batch_size=args.batch_size,
        datafile=datafile,
        shuffle_dataset=args.shuffle_dataset,
        num_workers=n_workers
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=make_checkpt_dir(args),
        save_top_k=1,
        monitor=args.criterion+'/validation',
        mode="min",
        filename=make_file_prefix(args)+'{pc_err/validation:.2e}',
        auto_insert_metric_name=False,
        save_last=False
        )

    model = MLP(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        consts_dict=consts_dict,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.results_dir, "TB_logs"),
        name='Method_'+str(args.method),
        default_hp_metric=True
    )

    logger.log_hyperparams(
        args
        )

    trainer = Trainer(
        logger=logger,
        gpus=avail_gpus,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        auto_lr_find=args.auto_lr_find,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        )

    trainer.fit(
        model,
        train_dl,
        val_dl
        )

    trainer.test(model, test_dl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--method", default=0, type=int)

    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--auto_lr_find", default=False, type=bool)
    parser.add_argument("--criterion", default='pc_err', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--which_spacing", default='both', type=str)
    parser.add_argument("--shuffle_dataset", default=True, type=bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=False, type=bool)
    parser.add_argument("--fast_dev_run", default=True, type=bool)
    args = parser.parse_args()

    main(args)

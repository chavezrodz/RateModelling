from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from misc.iterators import get_iterators
from misc.utils import make_file_prefix, make_checkpt_dir
from misc.load_model import load_model
import os


def main(args):
    n_workers = 8
    utilities.seed.seed_everything(seed=args.seed, workers=True)

    (train_dl, val_dl, test_dl), consts_dict = get_iterators(
        method=args.method,
        dataset=args.proj_dir,
        datapath=args.data_dir,
        results_path=args.results_dir,
        which_spacing=args.which_spacing,
        batch_size=args.batch_size,
        shuffle_dataset=args.shuffle_dataset,
        num_workers=n_workers,
        val_samp=args.val_sample
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

    input_dim = 3 if args.proj_dir == 'rate_integrating' else 4

    model = load_model(args, saved=False)

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
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--method", default=3, type=int)

    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--auto_lr_find", default=True, type=bool)
    parser.add_argument("--criterion", default='pc_err', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--which_spacing", default='both', type=str)
    parser.add_argument("--proj_dir", default='rate_modelling', type=str,
                        choices=['rate_modelling', 'rate_integrating'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--data_dir", default='../datasets', type=str)
    parser.add_argument("--shuffle_dataset", default=True, type=bool)
    parser.add_argument("--val_sample", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)

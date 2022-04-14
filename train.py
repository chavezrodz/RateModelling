from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from iterators import data_iterators
from MLP import MLP


def main(args, avail_gpus):
    datafile = 'method_'+str(args.method)+'.csv'
    utilities.seed.seed_everything(seed=args.seed)

    train, val, test = data_iterators(
        batch_size=args.batch_size,
        datafile=datafile,
        shuffle_dataset=args.shuffle_dataset
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath="Results/saved_models",
        save_top_k=1,
        monitor="valid/"+args.criterion,
        mode="min",
        filename='epoch={epoch}_val_pc_err={valid/pc_err:.2e}',
        auto_insert_metric_name=False,
        save_last=False
        )

    model = MLP(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,

        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )

    logger = TensorBoardLogger(
        save_dir='Results/TB_logs',
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
        log_every_n_steps=10
        )

    trainer.fit(
        model,
        train,
        val
        )


if __name__ == '__main__':
    AVAIL_GPUS = 0

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=6, type=int)
    parser.add_argument("--method", default=0, type=int)

    parser.add_argument("--shuffle_dataset", default=True, type=bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--criterion", default='pc_err', type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    args = parser.parse_args()

    main(args, AVAIL_GPUS)

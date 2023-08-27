from argparse import ArgumentParser
from misc.load_model import load_model
from misc.Datamodule import DataModule
import os


def main(args):
    for proj_dir in ['rate_modelling', 'rate_integrating']:
        for method in range(4):
            data_dir = args.data_dir
            results_dir = args.results_dir
            which = args.which_spacing
            h_dim = args.hidden_dim
            n_layers = args.n_layers

            dm = DataModule(method, proj_dir, data_dir, results_dir, which,
                            batch_size=8)
            model = load_model(method, h_dim, n_layers, proj_dir, dm,
                               saved=True, results_dir=results_dir)

            dm.setup(stage='test')
            for batch in dm.test_dataloader():
                x = batch[0]
                input_example = x[:4]
                break

            compiled_dir = os.path.join(
                results_dir,
                "compiled_models",
                proj_dir,
                )
            os.makedirs(compiled_dir, exist_ok=True)

            model.to_torchscript(
                file_path=os.path.join(compiled_dir, f'Method_{method}.pt'),
                example_inputs=input_example
                )


if __name__ == '__main__':
    parser = ArgumentParser()
    # Managing params

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--data_dir", default='../datasets', type=str)
    parser.add_argument("--which_spacing", default='both', type=str)

    # Model Params
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=8, type=int)

    args = parser.parse_args()

    main(args)

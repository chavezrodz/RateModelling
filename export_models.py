from argparse import ArgumentParser
from misc.load_model import load_model
from misc.Datamodule import DataModule
import os


def main(args):
    for method in range(4):
        print(f"Compiling {args.proj_dir} method {method}")
        results_dir = args.results_dir

        dm = DataModule(
            method, args.proj_dir, args.data_dir, args.results_dir,
            args.which_spacing)
        model = load_model(
                method, args.hidden_dim, args.n_layers, args.proj_dir,
                dm, saved=True, results_dir=args.results_dir,
                spacing=args.which_spacing)

        dm.setup(stage='test')
        for batch in dm.test_dataloader():
            x = batch[0]
            input_example = x[:4]
            break

        compiled_dir = os.path.join(
            results_dir,
            "compiled_models",
            args.proj_dir,
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
    parser.add_argument("--which_spacing", default='linspaced', type=str)

    parser.add_argument("--proj_dir", default='rate_integrating', type=str,
                        choices=['rate_modelling', 'rate_integrating'])

    # Model Params
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=4, type=int)

    args = parser.parse_args()

    main(args)

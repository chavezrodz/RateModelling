from misc.norms import normalize_vector_integrating, normalize_vector_modelling
from misc.Wrapper import Wrapper
from misc.MLP import MLP
import os


def load_model(args, dm, saved):
    h_dim = args.hidden_dim
    n_layers = args.n_layers
    method = args.method
    results_dir = args.results_dir

    core_model = MLP(
        input_dim=dm.input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        output_dim=1
        )

    if args.proj_dir == 'rate_modelling':
        normalize_func = normalize_vector_modelling
    else:
        normalize_func = normalize_vector_integrating

    if saved:
        pc_err = args.pc_err
        model_file = f'M_{method}_n_layers_{n_layers}_hid_dim_{h_dim}'
        model_file += f'_val_pc_err={pc_err}.ckpt'
        model_path = os.path.join(
            results_dir, args.proj_dir, "saved_models",
            f'Method_{method}', model_file
            )

        model = Wrapper.load_from_checkpoint(
            checkpoint_path=model_path,
            core_model=core_model,
            consts_dict=dm.consts_dict,
            normalize_func=normalize_func,
            )
    else:
        model = Wrapper(
            core_model=core_model,
            consts_dict=dm.consts_dict,
            normalize_func=normalize_func,
            criterion=args.criterion,
            lr=args.lr,
            amsgrad=args.amsgrad
        )

    return model

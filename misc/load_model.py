from misc.norms import normalize_vector_integrating, normalize_vector_modelling
from misc.Wrapper import Wrapper
from misc.MLP import MLP
import numpy as np
import glob
import os


def load_model(method, h_dim, n_layers, proj_dir, dm, saved,
               results_dir=None, amsgrad=None, criterion=None, lr=None
               ):

    core_model = MLP(
        input_dim=dm.input_dim,
        hidden_dim=h_dim,
        n_layers=n_layers,
        output_dim=1
        )

    if proj_dir == 'rate_modelling':
        normalize_func = normalize_vector_modelling
    else:
        normalize_func = normalize_vector_integrating

    if saved:
        model_file = f'M_{method}_n_layers_{n_layers}_hid_dim_{h_dim}_val_pc_'
        model_file = os.path.join(
            results_dir, proj_dir, "saved_models",
            f'Method_{method}', model_file
            )
        matching_models = glob.glob(model_file+'*')
        tmp_models = [
            float(match.split('=')[1].strip('.ckpt'))
            for match in matching_models
            ]
        idx = np.argmin(tmp_models)
        print(f"""
    Loading {proj_dir} Model {method} with {tmp_models[idx]*100:3}% error
        """)
        model_file = matching_models[idx]

        model = Wrapper.load_from_checkpoint(
            checkpoint_path=model_file,
            core_model=core_model,
            consts_dict=dm.consts_dict,
            normalize_func=normalize_func,
            )
    else:
        model = Wrapper(
            core_model=core_model,
            consts_dict=dm.consts_dict,
            normalize_func=normalize_func,
            criterion=criterion,
            lr=lr,
            amsgrad=amsgrad
        )
    return model

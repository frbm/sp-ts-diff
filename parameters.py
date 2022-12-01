import torch


params = {
    'channels': 64,
    'residual_layers': 24,
    'block_type': 'conv',
    'dilation_cycle_length': 10,

    'noise_schedule': torch.linspace(1e-4, 5e-1, 100),
    'inference_noise_schedule': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.2, 0.5],

    'learning_rate': 8e-4,
    'max_grad_norm': None,

    'saved_models_path': './saved_models',
}


class ParamSet:
    def __init__(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])


params = ParamSet(params)

import numpy as np
import torch


class DiffWaveInfer:
    def __init__(self, model, load_path=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.device = device

        if load_path is not None:
            self.load_model(load_path)

    def infer(self, input_noise, starting_date):
        """
        fast sampling
        :param input_noise:
        :param starting_date:
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            training_noise_schedule = np.array(self.model.params.noise_schedule)
            inference_noise_schedule = np.array(self.model.params.inference_noise_schedule)

            alpha = 1 - training_noise_schedule
            alpha_cum = np.cumprod(alpha)

            eta = inference_noise_schedule
            gamma = 1 - eta
            gamma_cum = np.cumprod(gamma)

            diff_steps = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if alpha_cum[t + 1] <= gamma_cum[s] <= alpha_cum[t]:
                        align = (alpha_cum[t] ** 0.5 - gamma_cum[s] ** 0.5) / (
                                alpha_cum[t] ** 0.5 - alpha_cum[t + 1] ** 0.5)
                        diff_steps.append(t + align)
                        break
            diff_steps = np.array(diff_steps, dtype=np.float32)

            series = input_noise[0].unsqueeze(0)
            for n in range(len(gamma) - 2, -1, -1):  # should be len(gamma)-1 not 2
                c1 = 1 / gamma[n] ** 0.5
                c2 = eta[n] / (1 - gamma_cum[n]) ** 0.5
                mu_fast = c1 * (series - c2 * self.model(series, starting_date,
                                                         torch.tensor([diff_steps[n]], device=series.device)).squeeze(
                    1))
                if n > 0:
                    noise = input_noise[n]
                    sigma_fast = ((1.0 - gamma_cum[n - 1]) / (1.0 - gamma_cum[n]) * eta[n]) ** 0.5
                    series += sigma_fast * noise + mu_fast
        return series

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime


class DiffWaveTrain:
    def __init__(self, model, dataset, optimizer, params, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.device = device
        self.params = params

        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_func = nn.L1Loss()

        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.grad_norm = None
        self.step = 0

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

    def train_step(self, series, starting_date):
        b, _, _ = series.size()
        series = series.to(self.device)
        starting_date = starting_date.to(self.device)
        self.noise_level = self.noise_level.to(self.device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [b], device=self.device)
            noise_scale = self.noise_level[t].unsqueeze(1).unsqueeze(2).repeat(1, series.size(1), series.size(1)).double()
            noise = torch.randn_like(series).double()

            noisy_series = torch.bmm(noise_scale**0.5, series) + torch.bmm((1. - noise_scale)**0.5, noise)
            noisy_series = noisy_series.float().to(self.device)

            prediction = self.model(noisy_series, starting_date, t)
            loss = self.loss_func(noise, prediction)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def train(self, max_steps):
        print(f'Max steps: {max_steps}.')
        print(f'Number of epochs: {max_steps // len(self.dataset)}')
        while self.step < max_steps:
            epoch_loss = 0
            for series, starting_date in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)+1}'):
                loss = self.train_step(series, starting_date)
                epoch_loss += loss.item()
                self.step += 1
            print(f'Mean epoch loss: {epoch_loss/len(self.dataset)}.')
            self.save_model()

    def save_model(self):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%Hh%m")
        torch.save(self.model.state_dict(), f'{self.params.saved_models_path}/model_{self.step}_{now}.pt')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

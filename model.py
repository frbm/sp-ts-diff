import torch
import torch.nn as nn
from math import sqrt, log


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.embedding = self.build_embedding(max_steps)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)
        self.act = nn.SiLU()

    def forward(self, diffusion_step):
        diffusion_step = diffusion_step.long()
        x = self.embedding[diffusion_step].to(diffusion_step.device)
        x = self.projection1(x)
        x = self.act(x)
        x = self.projection2(x)
        x = self.act(x)
        return x

    @staticmethod
    def build_embedding(max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / 63.0)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class DateEncoding(nn.Module):
    """Positional encoding for dates"""
    def __init__(self, dim, max_len=5000):
        super(DateEncoding, self).__init__()

        self.dim = dim
        self.max_len = max_len

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(log(15000.0) / dim))  # 15000 > number of date points
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)].requires_grad_(False).to(x.device)
        return x

    def remake_pe(self, starting_date):
        device = self.pe.device
        pes = torch.zeros(len(starting_date), self.dim, self.max_len)
        for i in range(len(starting_date)):
            sd = starting_date[i]
            pe = torch.zeros(self.max_len, self.dim)
            position = torch.arange(sd, sd + self.max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.dim, 2) * -(log(15000.0) / self.dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.transpose(0, 1)
            pes[i] = pe.unsqueeze(0)
        pes.to(device)
        self.register_buffer("pe", pes)


class ResidualBlock(nn.Module):
    def __init__(self, channels, block_type, block_param):
        """
        Residual Block for the
        :param channels:
        :param block_type: 'conv' or 's4'
        :param block_param: dilation parameter if block_type is 'conv', size if block_type is 's4'
        """
        super().__init__()
        if block_type == 'conv':
            self.block_op = nn.Conv1d(channels, 2*channels, 3, padding=block_param, dilation=block_param)
        elif block_type == 's4':
            pass
        else:
            raise NotImplementedError("'block_type' must be 'conv' or 's4'.")

        self.diffusion_projection = nn.Linear(512, channels)
        self.output_projection = nn.Conv1d(channels, 2*channels, 1)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.block_op(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.input_conv = nn.Conv1d(6, params.channels, 1)
        self.date_encoding = DateEncoding(dim=params.channels)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.channels, params.block_type, 2**(i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_conv = nn.Conv1d(params.channels, params.channels, 1)
        self.output_conv = nn.Conv1d(params.channels, 6, 1)
        self.relu = nn.ReLU()
        # nn.init.zeros_(self.output_conv.weight)

    def forward(self, x, starting_date, diffusion_step):
        self.date_encoding.remake_pe(starting_date)
        x = self.input_conv(x)
        x = self.relu(x)
        # x = self.date_encoding(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        skip = None
        for layer in self.residual_layers:
            x, skip_con = layer(x, diffusion_step)
            skip = skip_con if skip is None else skip_con + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_conv(x)
        x = self.relu(x)
        x = self.output_conv(x)
        return x

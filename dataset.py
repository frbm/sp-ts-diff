import torch
import pandas as pd
from torch.utils.data import Dataset


class TemperatureDataset(Dataset):
    def __init__(self, path, series_length=3288, max_date=None):
        super(TemperatureDataset, self).__init__()

        df = pd.read_csv(path, index_col='dates')
        df_tensor = torch.tensor(df.transpose().values)
        self.df = torch.cat([torch.arange(0, len(df)).unsqueeze(0), df_tensor], axis=0)
        if max_date is not None:
            self.df = self.df[:, :max_date]

        self.series_length = series_length

    def __getitem__(self, item):
        output = self.df[:, item:(item+self.series_length)]
        return output[1:], output[0][0].long()

    def __len__(self):
        return len(self.df[0]) - self.series_length + 1

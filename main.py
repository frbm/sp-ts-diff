from torch.utils.data import DataLoader
from model import *
from parameters import *
from train import *
from infer import *
from dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DiffWave(params).to(device)
print('Number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

tr_dataset = TemperatureDataset('./data/df_train.csv', series_length=300, max_date=7700)
tr_dataloader = DataLoader(tr_dataset, batch_size=16, shuffle=True)

trainer = DiffWaveTrain(model, tr_dataloader, optimizer, params, fp16=False)
trainer.train(max_steps=100000)

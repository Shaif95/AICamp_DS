import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#given the last 5 days of (Open,High,Low,Close,Volume) predict the next day's Close price
class StockDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        #remove the Date
        self.data = self.data.drop(columns=['Date'])
        #remove Volume
        self.data = self.data.drop(columns=['Volume'])
        self.transform = transform
    
    def __len__(self):
        #remeber, each sample is 5 days of data
        return len(self.data) - 6 

    #retun an input and output tensor
    def __getitem__(self, idx):
        #get the last 5 days of data
        input_tensor = torch.tensor(self.data.iloc[idx:idx+5].values)
        output_tensor = torch.tensor(self.data.iloc[idx+5]['Close'])
        return input_tensor, output_tensor
    
Data = StockDataset(csv_file = "./data/stocknet-dataset-master/price/raw/AAPL.csv")
data = DataLoader(Data, batch_size=1, shuffle=True, num_workers=0)

#make the LSTM model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=10, batch_first=True) 
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 1)
    
    def forward(self, x):
        h0 = torch.zeros(10, x.size(0), 128).cuda()
        c0 = torch.zeros(10, x.size(0), 128).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.relu(hn[-1])
        x = self.fc1(x)
        x = x.view(-1)
        return x
    
net = Net().cuda()

#make the loss function and optimizer
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

#train the model
for epoch in range(200):
    running_loss = 0.0
    for i, (input_sample, output_sample) in enumerate(data):
        #zero the parameter gradients
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = net(input_sample.float().cuda())
        loss = criterion(outputs, output_sample.float().cuda())
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, loss.item()))
            #print a sample
            running_loss = 0.0
        
print('Finished Training')

#save the model
PATH = './stock_net.pth'
torch.save(net.state_dict(), PATH)



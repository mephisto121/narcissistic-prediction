# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:59:01 2021

@author: LENOVO
"""

""" narcissistic prediction """

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import sqrt
from sklearn.utils import shuffle

data = pd.read_csv('path')
data = shuffle(data)
data = data.apply(pd.to_numeric)
data = data.apply(pd.to_numeric)

Y = data_v[:, 0:1]
X = data_v[:, 1:]

Y = torch.Tensor(Y).float()
X = torch.Tensor(X).float()

xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, test_size = 0.3, random_state = 42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(43, 90)
        self.fc2 = nn.Linear(90, 90)
        self.fc3 = nn.Linear(90, 40)
        self.fc4 = nn.Linear(40, 10)
        self.fc5 = nn.Linear(10, 1)
        self.l1 = nn.ReLU()
        self.l2 = nn.ReLU()
        self.l3 = nn.ReLU()
        self.l4 = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.l1(x)
        x = self.fc2(x)
        x = self.l2(x)
        x = self.fc3(x)
        x = self.l3(x)
        x = self.fc4(x)
        x = self.l4(x)
        x = self.fc5(x)
        
        return x
    
net = Net()

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.002)

epoch = 6000

for i in range(epoch):
    
    optimizer.zero_grad()
    y_pred  = net(xtrain)
    loss1 = loss(y_pred, ytrain)
    loss1.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        with torch.no_grad():
            
            y_predv = net(xvalid)
            y_predv = y_predv.round()
            print(sqrt(metrics.mean_squared_error(y_predv, yvalid)), metrics.r2_score(y_predv, yvalid),loss1.item())
    
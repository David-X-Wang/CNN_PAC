# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:35:52 2021

@author: David
"""

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder



def DWdataloader(DF2laod):
    
    df = pandas.read_csv(DF2laod)
    
    def Encoder(df):
              columnsToEncode = list(df.select_dtypes(include=['category','object']))
              le = LabelEncoder()
              for feature in columnsToEncode:
                  try:
                      df[feature] = le.fit_transform(df[feature])
                  except:
                      print('Error encoding '+feature)
              return df      
    
    EncodedData = Encoder(df)
    
    class MakeDataSet():  
                      
        def __init__(self):
            xy = EncodedData.values
            self.n_samples = xy.shape[0]
            self.y = torch.from_numpy(xy[:,0])
            self.x = torch.from_numpy(xy[:,1:])
            
        def __getitem__(self,index):
            return self.x[index], self.y[index]
            
        def __len__(self):        
            return self.n_samples
        
    EncodedData = MakeDataSet()
    return EncodedData

data2load = 'D:/PythonProject/functional/test_dataframe.csv'    
mydataset = DWdataloader(data2load)


num_batch = 20
dataloader = DataLoader(dataset=mydataset, batch_size=num_batch, shuffle = True)

dataiter = iter(trainloader)

data = dataiter.next()

feature,label = data

#%% Trainning

num_epoch = 200
total_samples = len(mydataset)
num_iter = int(np.ceil(total_samples/num_batch))
input_size = 7
output_size = 1
model = nn.Linear(input_size, output_size)


class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out
    
    
for epoch in range(num_epoch):
    for i , (features, labels,) in enumerate(trainloader):
    # forword 
      inputs, labels = torch.tensor(features), torch.tensor(labels)
         # Forward pass and loss
      y_predicted = model(inputs)
      loss = criterion(y_predicted, labels)
    
    # Backward pass and update
      loss.backward()
      optimizer.step()

    # zero grad before new step
      optimizer.zero_grad()

      if (epoch+1) % 10 == 0:
         print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
      
      
      
      
      
      
      
      print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')


# for epoch in range(2):
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs
#         inputs, labels = data

#         # wrap them in Variable
#         inputs, labels = tensor(inputs), tensor(labels)

#         # Run your training process
#         print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')

num_test =int(.15 * total_samples)
num_vali =int(.15 * total_samples)
num_train = total_samples-num_test-num_vali

train,vali,test =torch.utils.data.random_split(mydataset,[716,20,20])


trainloader = DataLoader(dataset=train, batch_size=num_batch, shuffle = True)

features,labels = trainloader

# 1) Model
# Linear model f = wx + b


# 2) Loss and optimizer
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
 



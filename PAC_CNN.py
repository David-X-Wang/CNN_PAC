# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:45:38 2021

@author: David
"""

from scipy.io import loadmat
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AHL = loadmat('AH-L-EEG.mat')['Thisfeature']

labels = loadmat('AH-L-labels.mat')['labels'].T

#%%

print('Total number of NaN entry is :',int(AHL[np.isnan(AHL)].shape[0]))
Nanind =[]
for i,k in enumerate(AHL):
    if np.isnan(k).any():
        Nanind.append(i)
AHL = np.delete(AHL,Nanind,0);
labels = np.delete(labels,Nanind,0);
print('After elimination, total number of NaN entry is :',int(AHL[np.isnan(AHL)].shape[0]))


Fs = 500

p = Pac(idpac=(1, 3, 0), f_pha=(1, 10, 1, .2), f_amp=(30, 100, 5, 1))

data = AHL
timevec = np.linspace(0,1.8,900)
xpac = p.filterfit(Fs, data)

xpac = np.transpose(xpac,(2,0,1))

p.comodulogram(xpac[0], cmap='Spectral_r', plotas='contour', ncontours=5,
               title=r'10hz phase$\Leftrightarrow$100Hz amplitude coupling',
               fz_title=14, fz_labels=13)

a = xpac.reshape(2,65,40);
phases = p.filter(Fs, data, ftype='phase')
amplitudes = p.filter(Fs, data, ftype='amplitude')
xpac.shape


#%%

# concatenate multi region QPCs as channels for CNN
# Then convert data and label to tensors
# QPCs = np.concatenate((AHL.reshape(AHL.shape[0],1,19,63),
#                        AHR.reshape(AHR.shape[0],1,19,63)),axis=1)

QPCs = torch.tensor(QPCs, dtype=torch.float)
labels = torch.tensor(labels,dtype=torch.float)


# Normalize QPC data (within the 'image')
mean = torch.tensor([0.5,0.5])
std = torch.tensor([0.5,0.5])
QPCs.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])




# concatenate QPC tensor and label tensors
QPCdataset  = TensorDataset(QPCs,labels)

train_dataset, test_dataset = torch.utils.data.random_split(QPCdataset, 
                                                            [round(QPCs.shape[0]*0.7), QPCs.shape[0]-round(QPCs.shape[0]*0.7)])

Train_set = DataLoader(train_dataset, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=True)


def testoutputsize(Train_set):
    dataiter = iter(Train_set)
    data = dataiter.next()
    feature,label = data
    feature.shape

    pool = nn.MaxPool2d(2,2)
    conv1= nn.Conv2d(2,8,3,[1,2],1) # in_chan,out_chan,kernel_size,stride_size,padding_size
    x = conv1(feature)
    print(x.shape)
    
    conv2 = nn.Conv2d(8,16,3,1,1)   
    x = conv2(x)
    x = pool(x)
    print(x.shape)
        
    conv3 = nn.Conv2d(16,32,2,[1,2])   
    x = conv3(x)
    x = pool(x)
    print(x.shape)
    
    


#%%

class QPCnet(nn.Module):
    def __init__(self,num_classes=2):
        super(QPCnet,self).__init__()
        
        
        self.conv1 = nn.Conv2d(2,8,3,[1,2],1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8,16,3,1,1)   
        self.conv3 = nn.Conv2d(16,32,2,[1,2])   
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self,x):
           
        x = (nn.functional.relu(self.conv1(x)))  
        x = self.pool(nn.functional.relu(self.conv2(x)))  
        x = self.pool(nn.functional.relu(self.conv3(x))) 
        x = x.view(-1, 32 * 4 * 4)           
        x = nn.functional.relu(self.fc1(x))              
        x = nn.functional.relu(self.fc2(x))              
        x = torch.sigmoid(self.fc3(x))                      
        
        
        return x
    
model = QPCnet().to(device)    

num_epoches = 1000;

learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
Loss_total = []
Acc_total = []
for epoch in range(num_epoches):
    for batch_ind, (data,label) in enumerate(Train_set):
        # use GPU if possible
        data = data.to(device=device)
        label = label.to(device=device)
        # forward 
        yhat = model(data)
        loss = criterion(yhat,label)
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        
        #gradient descent
        optimizer.step()
        
        _, predictions = yhat.max(1)
        num_correct = (predictions.reshape(predictions.shape[0],1) == label).sum()
        num_samples = predictions.size(0)
        accuracy_training = int(num_correct)/num_samples*100
        #if (batch_ind+1) % 8 == 0:

    print (f'Epoch [{epoch+1}/{num_epoches}], Loss: {loss.item():.4f}, Acc: {accuracy_training:.4f}')
    Loss_total.append(loss)     
    Acc_total.append(accuracy_training)
    
print('Finished Training')
PATH = './QPC_cnn.pth'
torch.save(model.state_dict(), PATH)        
        
        
        
        
        #%%
        
        
        
        
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions.reshape(predictions.shape[0],1) == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()        
        
check_accuracy(test_loader, model)        
        
        
        
#%%

plt.figure()
plt.plot(Loss_total) 
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')       
plt.savefig('Loss.jpg', format ='jpg')



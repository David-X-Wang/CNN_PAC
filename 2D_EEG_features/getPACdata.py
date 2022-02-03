
"""
Created on Wed Feb 10 15:45:38 2021

@author: David
"""

from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort



# load QPC raw data and labels
AHL = loadmat('/work/TIBIR/s201302/CNN/EEG_data/AH-L-EEG.mat')['Thisfeature']
labels = loadmat('/work/TIBIR/s201302/CNN/EEG_data/AH-L-labels.mat')['labels'].T

print('Total number of NaN entry is :',int(AHL[np.isnan(AHL)].shape[0]))

Nanind =[]
for i,k in enumerate(AHL):
    if np.isnan(k).any():
        Nanind.append(i)
    if (k==0).sum()==64*64:
        Nanind.append(i)

AHL = np.delete(AHL,Nanind,0)
labels = np.delete(labels,Nanind,0);

labels = labels.reshape(len(labels))


# num_recalled =(labels==1).sum();

# AHL = AHL[:num_recalled*2,:,:]
# labels = labels[:num_recalled*2]
print('After elimination, total number of NaN entry is :',int(AHL[np.isnan(AHL)].shape[0]))

print('Number of SuE trials is: '+str((labels==1).sum()))
print('Number of UnsuE trials is: '+str((labels==0).sum()))
#%%
Fs = 500
p = Pac(idpac=(2, 1, 3), f_pha=(1, 10, 1, .2), f_amp=(30, 100, 5, 1))
timevec = np.linspace(0,1.8,900)

AHL_recalled = AHL[np.where(labels==1)]
AHL_Nonrecalled = AHL[np.where(labels==0)]

PAC = np.empty((65,40,1))
AHLsplit = np.array_split(AHL,200)
for i,k in enumerate(AHLsplit):
    pac = p.filterfit(Fs, k,verbose=None)             
    PAC = np.concatenate((PAC,pac),axis=2)
    
    
PAC = np.delete(PAC,0,2)


# pac_recalled = p.filterfit(Fs, AHL_recalled)
# pac_nonrecalled = p.filterfit(Fs, AHL_Nonrecalled)

# pac_recalled = pac_recalled.transpose((2,0,1))
# pac_nonrecalled = pac_nonrecalled.transpose((2,0,1))

# pac_total = np.concatenate((pac_recalled,pac_nonrecalled))

# p.comodulogram(PAC[:,:,:20000], cmap='Spectral_r', plotas='contour', ncontours=5,
#                 title=r'10hz phase$\Leftrightarrow$100Hz amplitude coupling',
#                 fz_title=14, fz_labels=13)

# p.comodulogram(PAC[:,:,-20000:], cmap='Spectral_r', plotas='contour', ncontours=5,
#                 title=r'10hz phase$\Leftrightarrow$100Hz amplitude coupling',
#                 fz_title=14, fz_labels=13)
# phases = p.filter(Fs, data, ftype='phase')
# amplitudes = p.filter(Fs, data, ftype='amplitude')

torch.save(PAC, 'AH-L-PAC-canolty.pt')
# torch.save(xpac, 'AH-L-PAC.pt')
#torch.save(labels, 'AH-L-labels.pt')  
# torch.load('file.pt')

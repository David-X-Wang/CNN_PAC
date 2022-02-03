# -*- coding: utf-8 -*-
"""
This script computes functional/behavirol measures for subjects.
The subject dataset is obtained by geteegdata.m in MATLAB and the .mat dataset
follows such convension: 
    {'Brain area'}
    {'Electrode * Events * time/sampes'} during successful encoding
    {'Electrode * Events * time/sampes'} during unsuccessful encoding
example dataset:
UTXXX = |  ['AH-L']       ['AH-R']     ['PH-L']   ['PH-R']  |
        | [2*200*900]   [5*200*900]  [6*200*900]   []       |
        | [2*150*900]   [5*150*900]  [6*150*900]   []       |
        
Supported functional measures : oscilitory power
                                phase-amplitude coulping (local PAC)

This script uses pandas.DataFrame structure to store computed resutls that can 
be easily transfered to machine leanring structures (e.g., dataloader for PyTorch)
        
@author: David Wang
Jan/31/2021
"""

from scipy.io import loadmat
import scipy.stats 
from scipy.signal import butter,lfilter,hilbert
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time




class subject():
    
    def __init__(self,RecCondition,measures,RegLabels):
        self.RecCondition = RecCondition
        self.measures = measures
        self.RegLabels = RegLabels
    
def MeanQPC(QPCdataset):
    RegionLength = QPCdataset.shape[1]
    meanQPC = []
    MemScore = QPCdataset[1,0].shape[1]/(QPCdataset[1,0].shape[1]+QPCdataset[2,0].shape[1])
    for k in range(0,RegionLength):
        MeanQPCRecalled = np.mean(QPCdataset[1,k],axis =1)
        MeanQPCNonRecalled = np.mean(QPCdataset[2,k],axis =1)
        deltaQPC = MeanQPCRecalled-MeanQPCNonRecalled
        t, p = scipy.stats.ttest_ind(QPCdataset[1,k], QPCdataset[2,k], axis = 1, equal_var=False, alternative = 'greater')
       # Data_dict[Headers[2+k]]=np.mean(QPCtype)
        meanQPC.append(np.mean(deltaQPC))
        # Data_dict[Headers[2+k]] = np.mean(t)

    return [MemScore]+meanQPC
    

subj_sessions =  ['UT021','UT027','UT030','UT044','UT047','UT048','UT049','UT050',
                  'UT052','UT056','UT058','UT060','UT063','UT064','UT067','UT068',
                  'UT069','UT071','UT074','UT083','UT084','UT090','UT113','UT114',
                  'UT117','UT122','CC013','CC014','CC015','CC016','CC017','UT008',
                  'UT009','UT011','UT013','UT018','UT020','UT021','UT034','UT037',
                  'UT039','UT077','UT079','UT081','UT111'];

#subj_sessions  = ['UT063','UT064','UT067','UT068']

conditions = ['Recalled','NonRecalled']
dataPath = 'D:/QPCProject/QPCdataset/'
QPCdataset = loadmat(dataPath + '/' + subj_sessions[0] +'.mat')['QPCdataset']

Headers = ['MemScore','AH-L','AH-R','PH-L',     #features and labels 
           'PH-R'] 
Data_dict = {key:[] for key in Headers}
import os.path
for k in range(len(subj_sessions)):
    file2load = dataPath + '/' + subj_sessions[k] +'.mat'
    if os.path.isfile(file2load):
        QPCdataset = loadmat(file2load)['QPCdataset']
    else:
        print(subj_sessions[0],'is not found in',dataPath)
        continue
    
    if QPCdataset.shape[0]<2:
        print('No behavioral data found in ',subj_sessions[k],' please check the dataset')
        continue  
    
    datarow = MeanQPC(QPCdataset)
    [Data_dict[key].append(datarow[ind]) for ind,key in enumerate(Headers)] 
    
    
    
df = pd.DataFrame(Data_dict)



def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 50)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('dataset features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticks(np.arange(0,len(labels)))
    ax1.set_xticklabels(labels,fontsize=9,minor=False)
    ax1.set_yticks(np.arange(0,len(labels)))
    ax1.set_yticklabels(labels,fontsize=9,minor=False)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()
    
correlation_matrix(df)




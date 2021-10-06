#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 22:49:53 2021

@author: tingtingzhao
"""

#%% import
import os,sys
import numpy as np
#import torch
#import rpy2
#wd = os.getcwd()
wd = '/Users/guangyu/OneDrive - University of Florida/Research/Projects/YKnock/YKnock_public_repo/'
os.chdir(wd)


sys.path.append(wd)
sys.path.append(wd+'/core_codes') # add codes to seach path

#from stat_modelY_coef import stat_modelY_coef
#from stat_modelY_maxlam import stat_modelY_maxlam
from sklearn import preprocessing
import knockpy
from DeepYknock import DeepYknock
from rpy2.robjects.packages import importr
knockoff = importr('knockoff')

## package used to log the model with dropout 


r = 400  # number of response
p = 20  
n = 200 # sample size
m = 50    # num of active
rho=0.2
zseed = 1
betaValue = 1.5
perc=0
bias=False
hidden_sizes=[32,32]
normalize=False
verbose=True
 
## sample the indices that are zero for each row
def sampleInd(p, perc, seed=1):
    ## p is the total number of elements
    ## perc is the percentage of elements that we would like to sample
    nInd = int(p * perc)
    np.random.seed(seed)
    indices = np.random.choice(p, size = nInd, replace=False)
    ## return the indices that are sampled from [0, p] without replacement
    return indices

mr = m
mp = p
#
np.random.seed(2021)
Ac = np.arange(1,mr+1)
Ic = np.arange(mr+1,r+1)
SigmaX = knockpy.dgp.AR1(p=p, rho=rho) # Stationary AR1 process with correlation 0.5
betaA = np.random.choice([-betaValue,betaValue],mr*mp).reshape(mr,mp)
beta = np.zeros([r,p])
beta[0:mr,0:mp]=betaA
if perc > 0.01:
    for i in range(mp):
        ind = sampleInd(p, perc, i)
        beta[i, ind] = 0
else:
    beta=beta
    

def result(S):
    fdr=len(S.difference(Ac))/max(len(S),1)
    power=len(S.intersection(Ac))/max(len(Ac),1) 
    return power,fdr



#%%

X = np.random.multivariate_normal(mean=np.zeros(p), cov=SigmaX, size=(n,))

err = np.random.normal(0, 1, [n,r])
    
linear=np.matmul(X,beta.transpose())   # n by r

Y = linear + err 

scaleX = preprocessing.scale(X)
scaleY = preprocessing.scale(Y)    

Yk = np.array(knockoff.create_second_order(Y,method='equi'))
scaleYk = preprocessing.scale(Yk)
scaleYfeatures = np.concatenate((scaleY, scaleYk), axis=1)
   

model1 = DeepYknock(scaleX,scaleYfeatures,num_epochs=100,hidden_sizes=hidden_sizes,lambda1=0.001, lambda2=0.001,normalize=normalize,bias=bias,initW=None,verbose=verbose)
model1.trainModel() 
S1 = model1.filter()
result(S1)                       
    


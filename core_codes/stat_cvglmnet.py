#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform glmnet with python pacakge glmnet
Seems very slow compare to R


Created on Sun Aug  4 20:38:56 2019

@author: guangyu
"""
import numpy as np
from glmnet import glmnet
from glmnetCoef import glmnetCoef
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
import scipy

def stat_cvglmnet(X, Y, Yk,nlambda=100):
    Yfeatures = np.concatenate((Y, Yk), axis=1)
    fit = glmnet(x=Yfeatures.copy(),y=X.copy(),family='mgaussian')
    ss=glmnetCoef(fit,s = scipy.float64([0.5]), exact = False)
    ss.shape
    ssd=np.asarray(ss)
    
    return W
    

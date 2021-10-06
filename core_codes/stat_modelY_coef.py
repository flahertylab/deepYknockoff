#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:38:56 2019

@author: guangyu
"""
import numpy as np
#from scipy.stats import nbinom
#from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()

def stat_modelY_coef(X, Y, Yk,nlambda=100):
    rstring="""
    stat_modelY_coef <- function(X, Y, Yk, family='mgaussian',generate_lambda=TRUE,nlambda=100) {
      # Standardize variables
      M = scale(cbind(Y,Yk))
        N = X
        n = nrow(N); p = ncol(N)
        r = ncol(M)/2
        orig = 1:r
        #parallel=FALSE
  
        # Compute statistics
        if (generate_lambda) {
          # Unless a lambda sequence is provided by the user, generate it
          print('generate lambda')
          lambda_max = max(abs(t(M) %*% N)) / n
          lambda_min = lambda_max / 2e3
          k = (0:(nlambda-1)) / nlambda
          lambda = lambda_max * (lambda_min/lambda_max)^k
        }
        else {
          lambda = NULL
        }
        cv.glmnet.fit <- glmnet::cv.glmnet(M,N,lambda=NULL,family='mgaussian',intercept=TRUE,
                                           standardize=F,standardize.response=F, parallel=FALSE, nfolds = 5)
  
        coeff <- coef(cv.glmnet.fit, s = "lambda.min") # list of p (2r+1) vectors
        betas = matrix(0,2*r,p)
        for(i in 1:length(coeff)){
          betas[,i]<-as.matrix(coeff[[i]])[-1]
        }
        Z<-apply(betas,1,function(ss){sum(ss^2)})
        #plot(Z)
        #W = pmax(Z[orig], Z[orig + r])
        #chi = sign(Z[orig] - Z[orig + p])#*(1-2*swap)
        #W = abs(Z[orig]) - abs(Z[orig+r])
        return(Z)
    }
    """
    stat_modelY_coef=robjects.r(rstring)
    W=np.array(stat_modelY_coef(X,Y,Yk,nlambda=nlambda))
    return W
    

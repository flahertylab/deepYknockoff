#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:38:56 2019

@author: tingtingzhao
"""
import numpy as np
#from scipy.stats import nbinom
#from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()

def stat_modelY_classification_coef(X, Y, nlambda=100):
    rstring="""
    stat_modelY_classification_coef <- function(X, Y,nlambda=500,standardize=TRUE,parallel=FALSE) {
      # Randomly swap columns of X and Xk
      #swap = rbinom(ncol(X),1,0.5)
      #swap.M = matrix(swap,nrow=nrow(X),ncol=length(swap),byrow=TRUE)
      #X.swap  = X * (1-swap.M) + Xk * swap.M
      #Xk.swap = X * swap.M + Xk * (1-swap.M)
      t1<-proc.time()
      M = as.matrix(Y)
      N = as.matrix(X)
      n = nrow(N); p = ncol(N)
      r = ncol(M)/2
      orig = 1:r
      #parallel=FALSE
      glmnet.fit <- glmnet::glmnet(M,N,nlambda = nlambda,family='multinomial',intercept=TRUE,
                                   type.measure = "class", standardize=F,standardize.response=F, parallel=FALSE, nfolds = 5)
      
      coeff <- glmnet.fit$beta
      #betas = matrix(0,2*r,length(coeff))
      first_nonzero <- function(x) match(T, abs(x) > 0) # NA if all(x==0)
      
      Zs=matrix(0,2*r,length(coeff))
      
      for(i in 1:length(coeff)){
        beta<-as.matrix(coeff[[i]])
        indices <- apply(beta, 1, first_nonzero)
        names(indices) <- NULL
        Zs[,i] = ifelse(is.na(indices), 0,  glmnet.fit$lambda[indices] * n)
      }
      Z = apply(Zs,1,mean)
      
      # W = abs(Z[orig]) - abs(Z[orig+r])
      # plot(W)
      # print((proc.time()-t1)[3])
      return(Z)
      #
    }
    """
    stat_modelY_classification_coef=robjects.r(rstring)
    Z = np.array(stat_modelY_classification_coef(X,Y, nlambda=nlambda))
    return Z
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 23:36:57 2021

@author: tingtingzhao
"""

import numpy as np
#from scipy.stats import nbinom
#from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()

def stat_modelY_maxlam(X, Y, Yk,nlambda=100):
    rstring = """
    stat_modelY_maxlam <- function(X, Y, Yk,generate_lambda=TRUE,nlambda=100,standardize=TRUE) {
      # Randomly swap columns of X and Xk
      #swap = rbinom(ncol(X),1,0.5)
      #swap.M = matrix(swap,nrow=nrow(X),ncol=length(swap),byrow=TRUE)
      #X.swap  = X * (1-swap.M) + Xk * swap.M
      #Xk.swap = X * swap.M + Xk * (1-swap.M)
      t1<-proc.time()
      M = scale(cbind(Y,Yk))
      if(standardize)
        N = scale(X)
      else 
        N = X
      n = nrow(N); p = ncol(N)
      r = ncol(M)/2
      orig = 1:r
      #parallel=FALSE
      
      # Compute statistics
      if (generate_lambda) {
        # Unless a lambda sequence is provided by the user, generate it
        #print('generate lambda')
        lambda_max = max(abs(t(M) %*% N)) / n
        lambda_min = lambda_max / 2e3
        k = (0:(nlambda-1)) / nlambda
        lambda = lambda_max * (lambda_min/lambda_max)^k
      }
      else {
        lambda = NULL
      }
      glmnet.fit <- glmnet::glmnet(M,N,lambda=NULL,family='mgaussian',intercept=TRUE,
                                         standardize=F,standardize.response=F)
      coeff <- glmnet.fit$beta
      beta1 <- as.matrix(coeff[[1]])  # coefficent for X1   2*r by nlambda
      beta2 <- as.matrix(coeff[[2]]) 
      
      first_nonzero <- function(x) match(T, abs(x) > 0) # NA if all(x==0)
      indices <- apply(beta1, 1, first_nonzero)
      names(indices) <- NULL
      Z = ifelse(is.na(indices), 0,  glmnet.fit$lambda[indices] * n)
      # indices <- apply(beta2, 1, first_nonzero)
      # names(indices) <- NULL
      # Z2 = ifelse(is.na(indices), 0,  glmnet.fit$lambda[indices] * n)
      
      #W = abs(Z[orig]) - abs(Z[orig+r])
      #plot(W)
      #print((proc.time()-t1)[3])
      return(Z)
      #return(list(W=W,chi=chi))
      #
    }
    """
    stat_modelY_maxlam=robjects.r(rstring)
    Z=np.array(stat_modelY_maxlam(X,Y,Yk,nlambda=nlambda))
    return Z
    

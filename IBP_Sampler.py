#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats
import math
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


def IBP(alpha,N):
    """ Indian Buffett Process"""
    k = alpha * N * 10
    Z = np.zeros((N,k))
    
    #initial customer
    d = np.random.poisson(alpha)
    Z[0,0:d] = 1

    k_new = d
    #Rest of the customers
    for i in range(1,N):
        for j in range(k_new):
            probability = np.sum(Z[0:i,j])/(i + 1)
            if probability > np.random.random():
                Z[i,j] = 1
        d = np.random.poisson(alpha/(i + 1))
        Z[i,k_new:k_new + d] = 1
        k_new += d
        
    return Z[:,0:k_new]


# In[3]:


def log_likelyhood(X, N,D,K,sigma_X,sigma_A,Z):
    M = Z.T @ Z + (sigma_X**2/sigma_A**2)*np.eye(K)
    part1 = N*D/2 * np.log(2*np.pi) + (N - K)*D*np.log(sigma_X) + K*D*np.log(sigma_A)+D/2*np.log(np.linalg.det(M))
    part2_inside = np.eye(N) - (Z @ np.linalg.inv(M) @ Z.T)
    part2 = -1/(2 * sigma_X**2) * np.trace(X.T @ part2_inside @ X)
    return part2 - part1


# In[4]:


def sampler(X,alpha,niter,epsilon,sigma_X,sigma_A,alpha_a_prior,alpha_b_prior,max_new):
    N = X.shape[0]
    D = X.shape[1]
    Z = IBP(alpha,N) # set inital Z
    K = Z.shape[1]
    K_values = np.zeros(niter)
    alpha_values = np.zeros(niter)
    Sigma_X_values = np.zeros(niter)
    Sigma_A_values = np.zeros(niter)
    HN = 0
    for i in range(1,N+1):
        HN += 1.0/i
    for runs in range(niter):
        for i in range(N):
            for j in range(K):
                #Sample Z given conditionals
                
                col_k_count = sum(Z[:,j]) - Z[i,j] #p(zik|z-ik) = 0 so we set to 0
                if col_k_count == 0:
                    Z[i,j] = 0
                    
                else:
                    Z[i,j] = 0
                    Z0_p = log_likelyhood(X,N,D,K,sigma_X,sigma_A,Z) + np.log(N - col_k_count)
                    Z[i,j] = 1
                    Z1_p = log_likelyhood(X,N,D,K,sigma_X,sigma_A,Z) + np.log(col_k_count)
                    L = Z1_p - Z0_p
                    if L > 40: #helps with overflow
                        Z[i,j] = 1
                    elif L < -40:
                        Z[i,j] = 0
                    elif np.exp(L)/(1 + np.exp(L)) > np.random.random():
                        Z[i,j] = 1
                    else:
                        Z[i,j] = 0
                        
            #Sample to see if new columns get added
            log_prob = np.zeros(max_new)
            a_N = alpha/N
            log_prob[0] = -a_N + log_likelyhood(X,N,D,Z.shape[1],sigma_X,sigma_A,Z)
            for new_ks in range(1,max_new):
                new_cols = np.zeros((N,new_ks))
                new_cols[i,:] = 1
                Z_new = np.hstack((Z,new_cols))
                #Poisson(alpha/n) * log likelyhood
                log_prob[new_ks] = new_ks*np.log(a_N) - a_N - np.log(math.factorial(new_ks)) + log_likelyhood(X,N,D,Z_new.shape[1],sigma_X,sigma_A,Z_new)
            #Convert log likelyhoods
            prob = np.exp(log_prob - max(log_prob))
            prob = prob/sum(prob)

            #Sample probabilites and add columns accordingly
            new_cols_add = list(np.random.multinomial(1,prob) == 1).index(1)
            col_k_count = np.sum(Z,axis = 0) - Z[i,:]
            if new_cols_add == 0:
                Z = Z[:,col_k_count != 0]
            else:
                newcols = np.zeros((N,new_cols_add))
                newcols[i,:] = 1
                Z = np.hstack((Z[:,col_k_count != 0],newcols))
            K = Z.shape[1]
        
        #Part2
        current_likelyhood = log_likelyhood(X,N,D,K,sigma_X,sigma_A,Z) 
        
        #Sigma_X
        sigma_X_new = sigma_X + np.random.uniform(-epsilon,epsilon)
        new_likelyhood = log_likelyhood(X,N,D,K,sigma_X_new,sigma_A,Z)
        if new_likelyhood - current_likelyhood >= 0:
            sigma_X = sigma_X_new
        elif np.exp(new_likelyhood - current_likelyhood) > np.random.random():
            sigma_X = sigma_X_new
        else:
            sigma_X = sigma_X
            
        #Sigma_A
        sigma_A_new = sigma_A + np.random.uniform(-epsilon,epsilon)
        new_log_likelyhood = log_likelyhood(X,N,D,K,sigma_X,sigma_A_new,Z)
        if new_likelyhood - current_likelyhood >= 0:
            sigma_A = sigma_A_new
        elif np.exp(new_likelyhood - current_likelyhood) > np.random.random():
            sigma_A = sigma_A_new
        else:
            sigma_A = sigma_A
         
        #Alpha
        alpha = np.random.gamma(alpha_a_prior + K,alpha_b_prior + 1/(1 + HN))
        
        K_values[runs] = K
        alpha_values[runs] = alpha
        Sigma_X_values[runs] = sigma_X
        Sigma_A_values[runs] = sigma_A
        print(runs,K,sigma_X)
    return(K_values,alpha_values,Sigma_X_values,Sigma_A_values,Z)


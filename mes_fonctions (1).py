# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import norm


def partie_positive(x):
    return (x + abs(x))/2.

def mu(x):
    return 0.1*(np.sqrt(np.exp(x)) -1.) -1./8.


def mvt_brownien(T, n):
    W= [0.]
    delta = 0.25
    for i in range(n+1):
        wk = W[i] + norm.rvs(delta**2*(T/n))
        W.append(wk)
    return W


def eval_Xt(Xo, sigm, T, n):
    X = [Xo]
    W = mvt_brownien(T, n)
    for i in range(n+1):
        Xk = X[i] + mu(X[i]) * (T/n) + sigm*(W[i+1] - W[i])
        X.append(Xk)
    return X[-1]
    
def g(x, K, Xo, sigm, T, n):
    Xt = eval_Xt( Xo, sigm, T, n ) 
    return partie_positive(np.exp(Xt) - K)
    
def moyenne( list_resu ):
    M = 0.
    for x in list_resu:
        M += x 
    return M/len(list_resu)
    
def moyenne_antithetique( list_resu, list_resu_m ):
    array_resu = (np.array(list_resu) + np.array(list_resu_m) )/2
    m1= moyenne( array_resu )
    return m1 
    
def variance(list_resu ):
    var = 0.
    mean = moyenne(list_resu)
    for x in list_resu:
        var += (x - mean)
    return var/len(list_resu)
    
def covariance(list_resu, list_resu2 ):
    var = 0.
    mean1 = moyenne(list_resu)
    mean2 = moyenne(list_resu2)
    for x,y in zip(list_resu, list_resu2):
        var += (x - mean1)*(y-mean2)
    return var/len(list_resu)

def variable_control(list_resu1, list_resu2):
    var_2 = variance(list_resu2)
    cov   = covariance(list_resu1, list_resu2)
    b_opt = cov/var_2
    g_val = np.array(list_resu1) - b_opt * np.array(list_resu2)
    return variance( g_val )
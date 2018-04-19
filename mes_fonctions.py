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
    
def methode_gradient (K, Xo, sigm, T, n, n_itermax, theta_0):
    n_iter = 1
    theta_n = theta_0
    res = 1
    while (res > 1.e-3 and n_iter < n_itermax):
        X = eval_Xt(Xo, sigm, T, n)
        boutexp = np.exp((1./sigm**2)*(-(1./n_iter)*theta_n*X - ((theta_n*1./n_iter)**2)*1./2))
        g_x = g(X + theta_n, K, Xo, sigm, T, n)
        theta_next = theta_n - 1./n_iter * (g_x**2)*boutexp
        res = abs(theta_next - theta_n)
        #import pdb
        #pdb.set_trace()
        print("Iteration {} res = {}, theta = {}".format(n_iter,res,theta_next))
        theta_n = theta_next
        n_iter = n_iter + 1
    return theta_n
    
def suiteT (beta, n, T):
     list_T = [0]
     for K in range(n):
         val = np.random.gamma(K+1,1./beta)
         list_T.append(np.min(np.array([val,T])))
     list_T.append(T)
     return list_T
         
def calcul_N ( list_T):
    Kmax  = 0
    for i, t in enumerate(list_T):
        if t < list_T[-1]:
            Kmax = i
    return Kmax
            
def psi (beta, T, K, Xo, sigm, n):
    list_T = suiteT(beta, n, T)
    X = eval_Xt(Xo, sigm, list_T[-2], n)
    Xp = eval_Xt(Xo, sigm, list_T[-1], n)
    N = calcul_N(list_T)
    if N == 0:
        dg = g(Xp, K, Xo, sigm, T, n)
    else:
        dg = g(Xp, K, Xo, sigm, T, n) - g(X, K, Xo, sigm, T, n)
            
    prod = 1
    W = mvt_brownien(T, n)
    for K in range(N):
        prod *= (mu(X) - mu(Xp))*(W[K] - W[K-1])/(sigm*beta)*(list_T[K]-list_T[K-1])
    return np.exp(beta*T)*dg*prod
    
    
    
def calcul_strat(n_strate):
    list_a = [ 1./k for k in range(1,n_strate+1)]
    return list_a[::-1]
    
def phiinv(x):
    return -np.log(1.-x)
    
def methode_stratification( n_strate, n_tirage ,K, Xo, sigm, T, n  ):
    list_a = calcul_strat(n_strate) 
    list_X = []
    for k in range(n_strate-1):
        X_strate = []
        for i in range(n_tirage):
            var = list_a[k] + (list_a[k+1]-list_a[k]) * np.random.uniform(size=1)
            X_strate.append(phiinv( var ))
        list_X.append(X_strate)

    E = 0        
    for k in range(n_strate-1):
        E_strate  =0 
        for i in range(n_tirage):
            E_strate += g(list_X[k][i], K, Xo, sigm, T, n )
        E += (list_a[k+1]-list_a[k]) * (1./n_tirage) * E_strate
    return E

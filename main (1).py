# -*- coding: utf-8 -*-

from mes_fonctions import g, moyenne, moyenne_antithetique
import numpy as np


tirage = np.random.normal(loc=0.,scale=1.,size=3000)
K = 1
T = 1.
n = 10
sigm = 0.5
Xo = 0.

results = []
results_m = []
for x in tirage:
    results.append( g(x,K, Xo, sigm, T, n) )
    results_m.append( g(-x,K, Xo, sigm, T, n) )
    
E_MC = moyenne(results)
E_ANTI = moyenne_antithetique(results, results_m)
    
print("Moyenne : {}".format(E_MC))
print("Moyenne antithetique: {}".format(E_ANTI))




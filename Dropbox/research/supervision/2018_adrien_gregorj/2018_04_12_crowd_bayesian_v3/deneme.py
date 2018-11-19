#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:27:29 2018

@author: zeynep
"""
import numpy as np
import scipy as sc
      
def get_KL_single(psB, psA):
    """
    ps stands for probability distribution of a SINGLE observable
    
    For discrete probability distributions P and Q with a single dimension, 
    the Kullback–Leibler  divergence from Q to P is defined to be
 
    D(P|Q) = sum{ P(i) log(Q(i)/P(i)) }

    Here, Q is the observed distribution of the dyad of interest
    p is the empirical cumulative dist
    
    Careful with the order of inputs
    """
    alpha = 10**(-10)
    # The following eliminates undefined values and zeros.
    # Though it also introduces some level of unstability, 
    # it is still the most commonly adopted solution
    # Note that np.stats.entropy can handle the zeros in the first term (by 
    # omitting them) but not the zeros in second term
    psA = alpha * np.ones(len(psA)) + (1-alpha) * np.array(psA)
    psB = alpha * np.ones(len(psB)) + (1-alpha) * np.array(psB)
    
    # the below is necessary in joint case
    arr = np.multiply ( psA, (np.log(np.divide(psB, psA))))
    div_psB2psA = -np.sum(arr)
    
    return div_psB2psA, arr

def get_KL_joint(pjB, pjA):
    """
    pj stands for a JOINT probability distribution (with one or more observables)
    
    For discrete probability distributions P and Q, where P and Q can be 
    decomposed into the following products (ie assuming independece)
        P = P_1 * P_2
        Q = Q_1 * Q_2
    then D(P,Q), ie distance from Q to P, becomes
        D(P,Q) = sum {P2 .* D(P1, Q1)} + sum{P1 .* D(P2, Q2)}

    Here I use .* to denote elementwise multiplication.
    
    If the number of dimensions increase, eg
        P = P_1 * P_2 * P_3
        Q = Q_1 * Q_2 * Q_3
    then,
        D(P,Q) = sum {P2 * P3 .* D(P1, Q1)} + sum{P1 .* P3 .* D(P2, Q2)}
                + sum{P2 .* P3 .* D(P3, Q3)}
    """
    div_pjB2pjA = 0
    for o1 in range(np.size(pjA,0)):
        
        temp, d_sub  = get_KL_single(pjB[o1], pjA[o1])
        
        temp = np.ones_like(d_sub)
        for o2 in range(np.size(pjA,0)):
            if o1 is not o2:
                print('{} {}'.format( o1, o2))
                temp = np.multiply(temp, pjA[o2]) # this is element-wise
                
        div_pjB2pjA = div_pjB2pjA + np.sum(np.multiply(d_sub, temp)) # since it is minus
        
 
    return div_pjB2pjA

def get_KL_joint_v2(pjB, pjA):
    """
    pj stands for a JOINT probability distribution (with one or more observables)
    
    For discrete probability distributions P and Q, where P and Q can be 
    decomposed into the following products (ie assuming independece)
        P = P_1 * P_2
        Q = Q_1 * Q_2
    then D(P,Q), ie distance from Q to P, becomes
        D(P,Q) = sum {P2 .* D(P1, Q1)} + sum{P1 .* D(P2, Q2)}

    Here I use .* to denote elementwise multiplication.
    
    If the number of dimensions increase, eg
        P = P_1 * P_2 * P_3
        Q = Q_1 * Q_2 * Q_3
    then,
        D(P,Q) = sum {P2 * P3 .* D(P1, Q1)} + sum{P1 .* P3 .* D(P2, Q2)}
                + sum{P2 .* P3 .* D(P3, Q3)}
    """
    div_pjB2pjA = 0
    for o1 in range(np.size(pjA,0)):
        
        temp, d_sub  = get_KL_single(pjB[o1], pjA[o1])
        
        temp = np.ones_like(d_sub)
        for o2 in range(np.size(pjA,0)):
            if o1 is not o2:
                print('{} {}'.format( o1, o2))
                temp = np.multiply(temp, pjA[o2]) # this is element-wise
                
        div_pjB2pjA = div_pjB2pjA + np.sum(np.multiply(d_sub, temp)) # since it is minus
        
 
    return div_pjB2pjA
#############################################################################

x = [[0.7, 0.1, 0.1, 0.1], [0.5, 0.1, 0.1, 0.3]]
y = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.2, 0.3]]

# from y0 to x0
arry02x0 = [\
            x[0][0]*np.log(y[0][0]/x[0][0]), \
            x[0][1]*np.log(y[0][1]/x[0][1]), \
            x[0][2]*np.log(y[0][2]/x[0][2]), \
            x[0][3]*np.log(y[0][3]/x[0][3])]

arry02x0_kl_manual = -np.sum( arry02x0 )
arry02x0_kl_auto, temp = get_KL_single(y[0], x[0])

print('arry02x0:' , arry02x0)
print('arry02x0_kl_manual:' , arry02x0_kl_manual)
print('arry02x0_kl_auto:' , arry02x0_kl_auto)

# from y1 to x1
arry12x1 = [\
            x[1][0]*np.log(y[1][0]/x[1][0]), \
            x[1][1]*np.log(y[1][1]/x[1][1]), \
            x[1][2]*np.log(y[1][2]/x[1][2]), \
            x[1][3]*np.log(y[1][3]/x[1][3])]

arry12x1_kl_manual = -np.sum( arry12x1 )
arry12x1_kl_auto, temp = get_KL_single(y[1], x[1])

print('arry12x1:' , arry02x0)
print('arry12x1_kl_manual:' , arry12x1_kl_manual)
print('arry12x1_kl_auto:' , arry12x1_kl_auto)

arry2s_joint_kl_manual = np.sum(np.multiply(x[1], arry02x0) + np.multiply(x[0], arry12x1))
arry2s_joint_kl_auto = get_KL_joint(y, x)

print('arry2s_joint_kl_manual:' , arry2s_joint_kl_manual)
print('arry2s_joint_kl_auto:' , arry2s_joint_kl_auto)
 
#get_KL_joint(x, y)   
    
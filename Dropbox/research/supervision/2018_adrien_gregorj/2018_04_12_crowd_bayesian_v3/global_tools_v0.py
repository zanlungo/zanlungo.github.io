#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:14:58 2018

@author: zeynep
"""

import numpy as np
      
def get_KL_div(q, p):
    """
    For discrete probability distributions P and Q, the Kullback–Leibler 
    divergence from Q to P is defined to be
 
    D(P|Q) = sum{ P(i) log(Q(i)/P(i)) }

    q is the observed distribution of the pair of interest
    p is the empirical cumulative dist
    
    Careful with the order of inputs
    """
    alpha = 10**(-10)
    # this eliminates the undefined values, zeros, but also makes it unstable...
    # there seems to be no better way
    q = alpha * np.ones(len(q)) + (1-alpha) * np.array(q)
    p = alpha * np.ones(len(p)) + (1-alpha) * np.array(p)

    div_q2p = -np.sum(p*(np.log(q) - np.log(p)))
    
    return div_q2p  


def get_JS_div(q, p):
    """
    For discrete probability distributions P and Q, the Jensen-Shannon
    divergence from Q to P is defined to be
 
    D(P|Q) = 0.5sum{ P(i) log(M(i)/P(i)) } + 0.5sum{ Q(i) log(M(i)/Q(i)) }

    """
    # midpoint of p and q
    m = (p+q)/2

    div_symmetric = 0.5*get_KL_div( m, p) + 0.5*get_KL_div( m, q)
    
    return div_symmetric 

def get_EMD(pdf1, pdf2):
    """
    Computes earth mover distance between two pdfs 
    The order of inputs does not matter
    """
   
    emd = []
    emd.append(0);
    
    if((np.sum(pdf1) - np.sum(pdf2)) > 2.220**(-10) ):
        print('sum(pdf1) = {0:.5f}  sum(pdf2) = {0:.5f} Make sure arrays are scaled'.format(\
              (np.sum(pdf1),  np.sum(pdf2))) )
        return 0
            
    for i  in range(0, len(pdf1)):
        emd.append(pdf1[i] + emd[i] - pdf2[i])
    
    emd = np.sum(np.abs(emd))
    
    return emd

def get_LL(query_hist, base_pdf):
   """
   Since longer trajectories yield smaller values, 
   for making it independent of the trajectory length, 
   I take the mean, not the sum.
   For avoiding convergence to zero, I work with log.
   """
   # normalize to 1
   # because we are not integrating here so it is not necessary
   # to account for the bin size.
   # the values to should sum to 1 for log to be fair.
   base_pdf = base_pdf / np.sum(base_pdf)
   # only nonzero values
   temp1 = [num for num in np.power(base_pdf, query_hist) if num]
  
   #LL = np.sum(np.log(temp1))
   LL = -np.product(temp1)
   return LL

def compute_distance(measure, train_pdf, train_hist, test_pdf, test_hist):
    if measure is 'KLdiv':
        return get_KL_div(test_pdf, train_pdf)
    elif measure is 'JSdiv':
        return get_JS_div(test_pdf, train_pdf)
    elif measure is 'EMD':
        return get_EMD(test_pdf, train_pdf)
    elif measure is 'LL':
        return get_LL(test_hist, train_pdf)

   
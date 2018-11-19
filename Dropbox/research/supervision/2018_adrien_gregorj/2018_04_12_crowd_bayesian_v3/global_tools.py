#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:14:58 2018

@author: zeynep
"""

import numpy as np

import preferences
from importlib import reload
reload(preferences)

import constants
from importlib import reload
reload(constants)

      
def get_KL_single(psB, psA):
    """
    KL divergence **FROM B TO A**
    
    ps stands for probability distribution of a SINGLE observable (univariate 
    random variable)
    
    For discrete univariate probability distributions P and Q, the 
    Kullback–Leibler divergence from Q to P is defined as:
        
        D(P|Q) = sum{ P(i) log(Q(i)/P(i)) }

    In the input seqeuce, psB is the observed distribution of the dyad of
    interest psA is the empirical cumulative dist
    
    Careful with the order of inputs!
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
    
    return div_psB2psA

def get_KL_joint(pjB, pjA):
    """
    pj stands for a JOINT probability distribution (with one or more observables)
    
    For discrete probability distributions P and Q, where P and Q can be 
    decomposed into the following products (ie assuming independece)
        P = P_1 * P_2
        Q = Q_1 * Q_2
    then D(P,Q), ie distance from Q to P, becomes simply
        D(P,Q) = D(P1, Q1) + D(P2, Q2)

    
    If the number of dimensions increase, eg
        P = P_1 * P_2 * P_3
        Q = Q_1 * Q_2 * Q_3
    then, 
        D(P,Q) = D(P1, Q1) + D(P2, Q2) + D(P3, Q3)
        
    In below function, the number of dimensions is simply the number of observables. 
    """
    div_pjB2pjA = 0
    for o1 in preferences.OBSERVABLES:
        div_psB2psA_sub  = get_KL_single(pjB[o1], pjA[o1])       
        div_pjB2pjA = div_pjB2pjA + div_psB2psA_sub
        
 
    return div_pjB2pjA

def get_JS_single(qs, ps):
    """
    JS divergence is symmetric. This function computes JS divergence 
    from B to A (or equivalently from A to B)

    For discrete probability distributions P and Q, the Jensen-Shannon
    divergence is defined to be
 
        D(P|Q) = 0.5 * sum{ P(i) log(M(i)/P(i)) } + 0.5sum{ Q(i) log(M(i)/Q(i)) }
    
    where M is the midpoint of P and Q

    """
    # midpoint of p and q
    m = (ps+qs)/2

    div_symmetric = 0.5*get_KL_single( m, ps) + 0.5*get_KL_single( m, qs)
    
    return div_symmetric 

def get_JS_joint(pjB, pjA):
    """
    pj stands for a JOINT probability distribution (with one or more observables)

    For discrete probability distributions P and Q, where P and Q can be 
    decomposed into the following products (ie assuming independece)
        P = P_1 * P_2
        Q = Q_1 * Q_2
    then D(P,Q), ie distance from Q to P, becomes simply
        D(P,Q) = D(Q1, P1) + D(Q2, P2)

    
    If the number of dimensions increase, eg
        P = P_1 * P_2 * P_3
        Q = Q_1 * Q_2 * Q_3
    then, 
        D(P,Q) = D(Q1, P1) + D(Q2, P2) + D(Q3, P3)
        
    In below function, the number of dimensions is simply the number of observables. 

    """
    div_symmetric_pjB2pjA = 0
    for o1 in preferences.OBSERVABLES:
        div_sub  = get_JS_single(pjB[o1], pjA[o1])       
        div_symmetric_pjB2pjA = div_symmetric_pjB2pjA + div_sub
        
    return div_symmetric_pjB2pjA

def get_EMD_single(pdfA, pdfB):
    """
    Computes earth mover distance between two pdfs of a SINGLE (the same kind 
    of observable). Obviously, it is symmetric.
    
    Careful that the pdf's are normalized to sum up to 1, and **NOT** the 
    integral of the pdf is 1.
    
    The order of inputs does not matter
    """
   
    emd = []
    emd.append(0);
    
    if((np.sum(pdfA) - np.sum(pdfB)) > 2.220**(-10) ):
        print('sum(pdf1) = {0:.5f}  sum(pdf2) = {0:.5f} Make sure arrays are scaled'.format(\
              (np.sum(pdfA),  np.sum(pdfB))) )
        return 0
            
    for i  in range(0, len(pdfA)):
        emd.append(pdfA[i] + emd[i] - pdfB[i])
    
    emd = np.sum(np.abs(emd))
    
    return emd

def get_EMD_joint(pjA, pjB):
    """
    Computes earth mover distance between two sets of pdfs. Therefore, I call 
    it joint. Obviously, it is symmetric and the order of inputs does not 
    matter.
    
    Note that I use pdf's scaling up to 1 (and  **NOT** the integral)
    
    But then, in order to have a value as independent as the number of bins, I 
    scale the components with the associated bin_size.
    
    Due to the assumtion of independence of obsevables, I sum up the divergence 
    along each dimension. 
    
    """
    div_symmetric_pjB2pjA = 0
    
    for o1 in preferences.OBSERVABLES:
        
        tempA = pjA[o1] / np.sum(pjA[o1])
        tempB = pjB[o1] / np.sum(pjB[o1])
        (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[o1]
        n_bins = np.abs(max_bound-min_bound) / bin_size
  
        div_sub  = get_EMD_single(tempA, tempB) / n_bins
        
        div_symmetric_pjB2pjA += div_sub

    return div_symmetric_pjB2pjA

def get_LL_single(query_hist, base_pdf):
   """
   Since longer trajectories yield smaller values, 
   for making it independent of the trajectory length, 
   I take the mean, not the sum.
   For avoiding convergence to zero, I work with log.
   """
   # normalize to 1
   # because we are not integrating here so it is not necessary
   # to account for the bin size.
   # the values should sum to 1 for log to be fair.
   base_pdf = base_pdf / np.sum(base_pdf)
   # only nonzero values
   temp1 = [num for num in np.power(base_pdf, query_hist) if num]
  
   LL = np.sum(np.log(temp1))
   #LL = -np.product(temp1)
   return LL

def get_LL_joint(test_hists, train_pdfs):
   """
   In multiple dimensions with independence assumption
   Careful with order of inputs
   Depending on use of log or not adjust he inilizalization and +/* below
   """
   LL = 0
   for o1 in preferences.OBSERVABLES:
       LL += get_LL_single(test_hists[o1], train_pdfs[o1])
       
   return LL

def dist2prob(distances):
    """
    This function takes as input the divergence of an query dyad to training 
    distributions; and it turns these values into some values which may be 
    treated as probability. Note that they rae not real probabilities. But they
    sum up to one and reflect the superioirty relation, ie a smaller divergence 
    is associated with a larger 'probability'. But the scale is not preserved, 
    ie if Div(P,Q1) is twice of Div(P,Q2), then Prob(Q2) is not twice of 
    Prob(Q1).
    
    Say the quesry distributions is represented with P, and the training 
    distributions are represented with Qn, where 1 <= n <= N. Then:
        Prob(P ~ Qn) = [1 / (N-1)] * [1 - Div(P, Qn)/DivT]
    where DivT is:
        DivT = sum{i=1}^{N} Div(P, Qi)
    
    Consider the following example:
        Div(P, Q1) = 1
        Div(P, Q2) = 2
        Div(P, Q3) = 3
        Div(P, Q4) = 4
    According the above definition:
        Prob(P ~ Q1) = 1/3 * [1 - 1/10] = 9/30
        Prob(P ~ Q2) = 1/3 * [1 - 2/10] = 8/30
        Prob(P ~ Q3) = 1/3 * [1 - 3/10] = 7/30
        Prob(P ~ Q4) = 1/3 * [1 - 4/10] = 6/30 
    """ 
    
    N = len(distances)
    div_tot = sum(distances[item] for item in distances)
    
    p_est = {k: (N-1)**-1 * (1-v/div_tot) for k, v in distances.items()}  
    
    return p_est

def compute_distance(measure, train_pdfs, train_hists, test_pdfs, test_hists):
    if measure is 'KLdiv':
        return get_KL_joint(test_pdfs, train_pdfs)
    elif measure is 'JSdiv':
        return get_JS_joint(test_pdfs, train_pdfs)
    elif measure is 'EMD':
        return get_EMD_joint(test_pdfs, train_pdfs)
    elif measure is 'LL':
        return get_LL_joint(test_hists, train_pdfs)

   
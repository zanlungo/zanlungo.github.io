#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:43:59 2018

@author: zeynep
"""
import numpy as np
import generic_tools
import time

# if the constants/preferences are not refreshed, take a long way and reload
from importlib import reload

import preferences
reload(preferences)

import constants
reload(constants)


"""

This function computes the f-values for various distributions

 Go to the following website and input, v1, v2 and f-value.
 It computes 1-p-value
 Carefull here!!!

 http:stattrek.com/online-calculator/f-distribution.aspx

"""

def get_edges(obs):
    """
    Compute the abscissa value to plot the PDF of the given observable parameter
    """
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    return np.arange(min_bound, max_bound, bin_size)

def get_fval(means):
    
    fi, sigmai, ni = [], [], []
    
    for c in preferences.CLASSES: 
        fi.append( np.mean(means[c]) )
        sigmai.append( np.std(means[c]) )
        ni.append( len(means[c]))
        
    
    N = np.sum(ni)
    fbar = np.sum(np.multiply(fi,  ni)) / N
    di = np.subtract(fi , fbar)
    SST = np.sum(np.multiply(ni , np.square(di)))
    SSE = np.sum(np.multiply(ni , np.square(sigmai)))
    
    nbar = len(fi)
    kd = nbar - 1 # v1: 1st deg of freedom
    Nkd = N - nbar # v1: 2nd deg of freedom
    
    MST = SST / kd # ask2pthis
    MSE = SSE / Nkd
    
    F_d = MST / MSE
    
    return F_d, kd, Nkd

if __name__ == "__main__":

    start_time = time.time()
    
    histograms1D, pdf1D, mean_pdfs = {}, {}, {}
    
    # initialize arrays (empty histograms, pdfs)
    for o in preferences.OBSERVABLES:
        histograms1D[o], pdf1D[o], mean_pdfs[o] = {}, {}, {}
        for c in preferences.CLASSES:
            histograms1D[o][c] = []
            pdf1D[o][c] = []
            mean_pdfs[o][c] = []
                    
                   
    data_fnames = generic_tools.get_data_fnames('data/classes/')

    for c in preferences.CLASSES: 
        
        for file_path in data_fnames[c]:
            
            data = np.load(file_path)
            data_A, data_B = generic_tools.extract_individual_data(data)
            obs_data = generic_tools.compute_observables(data_A, data_B)
            
            for o in preferences.OBSERVABLES:
                
                edges = get_edges(o)
                
                temp_hist = generic_tools.compute_histogram_1D(o, obs_data[o])
                temp_pdf = generic_tools.compute_pdf(o, temp_hist)
                
                histograms1D[o][c].append(temp_hist )
                pdf1D[o][c].append( temp_pdf )
                
                mean_pdfs[o][c].append( np.average(edges, weights=temp_pdf) )
            
    print('Obs\tF_d\tv1\tv2')
    print('-------------------------------')
    for o in preferences.OBSERVABLES:
        F_d, v1, v2 = get_fval(mean_pdfs[o])
        print('{}\t{:2.3f}\t{}\t{}'.format(o, F_d, v1, v2))
        
    print('\n**Check this page for computing p-value from f-statistics**')
    print('http:stattrek.com/online-calculator/f-distribution.aspx ')
    elapsed_time = time.time() - start_time
    print('\nTime elapsed  %2.2f sec' %elapsed_time)
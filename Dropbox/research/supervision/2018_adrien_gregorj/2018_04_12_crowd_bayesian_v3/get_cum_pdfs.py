#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:12:17 2018

@author: zeynep
"""
import generic_tools
from importlib import reload
reload(generic_tools)

import constants
from importlib import reload
reload(constants)

# if the constants/preferences are not refreshed, take a long way and reload
import preferences
from importlib import reload
reload(preferences)

import matplotlib.pyplot as plt
import numpy as np
import time

def get_edges(obs):
    """
    Compute the abscissa value to plot the PDF of the given observable parameter
    """
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs]
    return np.arange(min_bound, max_bound, bin_size)

def  plot_pdf(pdfs):
    plt.rcParams['grid.linestyle'] = '--'
    for o in preferences.OBSERVABLES:
        plt.figure()
        edges = get_edges(o)
        for c in preferences.CLASSES_RAW:
            plt.plot(edges, pdfs[o][c], label=constants.TRADUCTION_TABLE[c], linewidth=3)
            
            data = np.array([edges, pdfs[o][c]])
            data = data.T
            datafile_path = 'figures/'+ constants.TRADUCTION_TABLE[c] + '_' +\
            constants.PARAM_NAME_TABLE[o].replace('\\','').replace('$','').replace('{','').replace('}','') + ".txt"
            with open(datafile_path, 'w+') as datafile_id:
            #here you open the ascii file
                np.savetxt(datafile_id, data, fmt=['%1.5f','%1.5f'])
            
        plt.xlabel('{}({})'.format(constants.PARAM_NAME_TABLE[o], constants.PARAM_UNIT_TABLE[o]))
        plt.ylabel('p({})'.format(constants.PARAM_NAME_TABLE[o]))
        plt.xlim(constants.PLOT_PARAM_TABLE[o])
        plt.legend()
        plt.grid()
        plt.show()
        
        year, month, day, hour, minute, second = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')

        figname = 'figures/'+\
        year +'_'+ month +'_'+ day +\
        '_'+ hour +'_'+  minute +'_'+ second + '.png'
        plt.savefig(figname)
        
        plt.pause(1)


if __name__ == "__main__":

    start_time = time.time()
    
    data_fnames = generic_tools.get_data_fnames('data/classes/')

    histograms1D = {}
    pdfs1D = {}
    # initialize empty histograms
    for o in preferences.OBSERVABLES:
        histograms1D[o], pdfs1D[o] = {}, {}
        for c in preferences.CLASSES:
            histograms1D[o][c] = generic_tools.initialize_histogram(o)
            
    # compute histograms for each class
    for c in preferences.CLASSES:   
        for file_path in data_fnames[c]:
            data = np.load(file_path)
            data_A, data_B = generic_tools.extract_individual_data(data)
            obs_data = generic_tools.compute_observables(data_A, data_B)
            for o in preferences.OBSERVABLES:
                histograms1D[o][c] += generic_tools.compute_histogram_1D(o, obs_data[o])
                
    for o in preferences.OBSERVABLES:
        for c in preferences.CLASSES:
            pdfs1D[o][c] = generic_tools.compute_pdf(o, histograms1D[o][c])
            
    plot_pdf(pdfs1D)
            
    elapsed_time = time.time() - start_time
    print('\nTime elapsed  %2.2f sec' %elapsed_time)
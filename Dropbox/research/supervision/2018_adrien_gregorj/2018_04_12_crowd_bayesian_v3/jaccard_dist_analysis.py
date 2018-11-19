#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:10:36 2018

@author: zeynep
"""
from os import listdir
import random

import bayesian_model_indep
from model import tools

from importlib import reload
import constants
reload(constants)
import preferences
reload(preferences)

import numpy as np
import time

def get_datasets(data_path, classes):
    """
    Get the dataset for the given classes
    """
    datasets = {}
    for c in classes:
        class_path = data_path + c +'/'
        class_set = [class_path + f for f in listdir(class_path) if 'threshold' in f]
        datasets[c] = class_set
    return datasets


def shuffle_data_set(datasets, train_ratio):
    """
    Randomly partition the datasets into training and testing sets
    """
    train_fnames = {}
    for c, dataset in datasets.items():
        n = len(dataset)
        n_train = round(train_ratio * n)
        shuffled_set = random.sample(dataset, n)
        train_fnames[c] = shuffled_set[:n_train]
    
    return train_fnames


def get_jaccard_dist(train_fnames):
    
    histograms, pdfs, joint_pdfs, jaccard_dist = {}, {}, {}, {}
    # initialize empty histograms
    for o in preferences.OBSERVABLES:
        histograms[o], pdfs[o] = {}, {}
        for c in preferences.CLASSES:
            histograms[o][c] = tools.initialize_histogram(o)
    # compute histograms for each classes
    obs_data_cum = {}
    for c in preferences.CLASSES:
        obs_data_cum[c] = {}
        for o in preferences.OBSERVABLES:
                 obs_data_cum[c][o] = []
        for file_path in train_fnames[c]:
            data = np.load(file_path)
            data_A, data_B = tools.extract_individual_data(data)
            obs_data = tools.compute_observables(data_A, data_B)
            for o in preferences.OBSERVABLES:
                histograms[o][c] += tools.compute_histogram(o, obs_data[o])
                obs_data_cum[c][o].extend(obs_data[o])
                
    for o in preferences.OBSERVABLES:
        for c in preferences.CLASSES:
            pdfs[o][c] = tools.compute_pdf(o, histograms[o][c])
            
 
            
    for c in preferences.CLASSES:
        joint_pdfs[c], jaccard_dist[c] = {}, {}
        for o1 in preferences.OBSERVABLES:
            joint_pdfs[c][o1], jaccard_dist[c][o1] = {}, {}
            for o2 in preferences.OBSERVABLES:
                joint_pdfs[c][o1][o2] = tools.compute_joint_pdf(tools.compute_joint_histogram(o1, obs_data_cum[c][o1], o2, obs_data_cum[c][o2]))
                joint_ent = tools.get_joint_ent(joint_pdfs[c][o1][o2], pdfs[o1][c], pdfs[o2][c])
                mutual_inf = tools.get_mutual_inf(joint_pdfs[c][o1][o2], pdfs[o1][c], pdfs[o2][c])
                # i should not need th follwoign wheck but all is nan 
                if mutual_inf is not 0:
                    jaccard_dist[c][o1][o2] = (joint_ent - mutual_inf) / joint_ent
                    
    return jaccard_dist

def get_mutual_inf(pj, q1, q2):
    """
    pj is the joint probabilty distribution.
    It is a pdf but I need to scale it to 1, otherwise bin size is not accounted
    So it looks like there is discrepancy.
    q1 and q2 are the individual distribution of the two variables.
    Similarly, they are scaled to 1.
    """
    q1 = q1 / np.sum(q1)
    q2 = q2 / np.sum(q2)
    
    mutual_inf = 0
    for i in range(0, len(q1)):
        for j in range(0, len(q2)):
            # only when all values are nonzero
            if 0 not in np.array([ pj[i,j], q1[i], q2[j] ]):
                mutual_inf += pj[i,j]*np.log(pj[i,j] / q1[i] / q2[j])    
    return mutual_inf
                
def get_joint_ent(pj, q1, q2):
    """
    Takes only joint pdf. Scaled to 1 as above.
    """
    q1 = q1 / np.sum(q1)
    q2 = q2 / np.sum(q2)
    
    joint_ent = 0
    for p in range(0, len(pj)):
        for q in range(0, len(pj[p])):
            #if not math.isnan(pj[p,q] * np.log(pj[p, q])):
            if 0 not in np.array([ pj[p,q], q1[p], q2[q] ]):
                joint_ent = joint_ent - pj[p,q] * np.log(pj[p, q])
    return joint_ent

def compute_joint_histogram(obs1, obs_data1, obs2, obs_data2):
    """
    here the assumption is that the obs_data arrays match in time. 
    """
    
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs1]
    n_bins = round((max_bound - min_bound) / bin_size) + 1
    edges1 = np.linspace(min_bound, max_bound, n_bins)
    
    (min_bound, max_bound, bin_size) = constants.HISTOG_PARAM_TABLE[obs2]
    n_bins = round((max_bound - min_bound) / bin_size) + 1
    edges2 = np.linspace(min_bound, max_bound, n_bins)
    
    histogram2D, edges1, edges2 = np.histogram2d(obs_data1, obs_data2, bins=(edges1, edges2))

    return histogram2D

def compute_joint_pdf(histogram2D):
    """
    Here I donot scale with bin size.
    """
    pdf_joint = histogram2D / np.sum(histogram2D)
    return pdf_joint

def display_stats(jaccard_dist):
    
    means, stds = {}, {}  
    for c in preferences.CLASSES:
        means[c], stds[c] = {}, {} 
        for o1 in preferences.OBSERVABLES:
            means[c][o1], stds[c][o1] = {}, {} 
            for o2 in preferences.OBSERVABLES:
                means[c][o1][o2], stds[c][o1][o2] = [], []

    
    
    for c in preferences.CLASSES:
        for o1 in preferences.OBSERVABLES:
            for o2 in preferences.OBSERVABLES:
                temp = []
                for e in range(preferences.N_EPOCH):
                    temp.append( jaccard_dist[e][c][o1][o2] )
                    
                means[c][o1][o2] = np.mean(temp)
                stds[c][o1][o2] = np.std(temp)
           
    print('\t', end='', flush=True)
    for o in preferences.OBSERVABLES:
        print('{}'.format(o), end='\t\t\t', flush=True)
    print('')
    
    for c in preferences.CLASSES:
        print('\n{}'.format(c))
        for o1 in preferences.OBSERVABLES:
            print('{}'.format(o1), end='\t', flush=True)
            for o2 in preferences.OBSERVABLES:
                print('{:.4f} pm {:.4f}'.format(means[c][o1][o2], stds[c][o1][o2]), end='\t', flush=True)
            print('')
        
                                
        
    
    


if __name__ == "__main__":

    start_time = time.time()
    
    bayesian_indep = bayesian_model_indep.BayesianEstimator()
    datasets = get_datasets('data/classes/', preferences.CLASSES)
    
    jaccard_dist = []
    
    print('Running for {} epochs'.format(preferences.N_EPOCH))
    
    for epoch in range(preferences.N_EPOCH):
        
        if epoch%10 is 0:
            print('')
        print("{}...".format(epoch), end ='', flush=True) 
        
        train_fnames = shuffle_data_set(datasets, preferences.TRAIN_RATIO)
        jaccard_dist.append(get_jaccard_dist(train_fnames=train_fnames))
        
    means, stds = display_stats(jaccard_dist)

    elapsed_time = time.time() - start_time
    print('\nTime elapsed  %2.2f sec' %elapsed_time)
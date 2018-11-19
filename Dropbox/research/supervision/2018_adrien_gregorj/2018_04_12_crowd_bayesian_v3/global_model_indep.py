import numpy as np
from scipy import spatial, stats
#from model import tools
import time


import operator

# if the constants/preferences are not refreshed, take a long way and reload
import generic_tools
from importlib import reload
reload(generic_tools)

import global_tools
from importlib import reload
reload(global_tools)

import preferences
from importlib import reload
reload(preferences)



class GlobalEstimator():
    """
    Global estimator does not make instantaneous decisions or compute 
    instantaneous propbabilities. 
    
    The decision/probability is computed after the entire trajecory is observed. 
    It bags all observations form a single dyad so temporal relation is not 
    utilized.
    """
    def __init__(self):
        self.train_pdfs = {}
        self.train_histograms = {}
        
        self.test_pdfs = {}
        self.test_histograms = {}
        
        self.data_fnames = generic_tools.get_data_fnames('data/classes/')
        
        
    def init_conf_mat(self):
        """
        Initialize cnfusion matrix or performance measures.
        """
        
        self.conf_mat = {'event_based':{'voting': {}, 'emp_probs':{}},\
                                 'trajectory_based':{'prob': {}, 'binary': {}, 'confidence':{}},\
                                 'collective': {'confidence': {}, 'binary':{}}}
        
        for class_gt in preferences.CLASSES:
                       
            self.conf_mat['event_based']['voting'][class_gt] = {}
            self.conf_mat['event_based']['emp_probs'][class_gt] = {}
            
            self.conf_mat['trajectory_based']['prob'][class_gt] = {}
            self.conf_mat['trajectory_based']['binary'][class_gt] = {}
            self.conf_mat['trajectory_based']['confidence'][class_gt] = []
            
            self.conf_mat['collective']['confidence'][class_gt] = {\
                         'cum_n_observations': 0, \
                         'cum_confidence': 0,\
                         'cum_confidence_sq': 0}
            self.conf_mat['collective']['binary'][class_gt] = {\
                         'n_suc': 0,\
                         'n_fail': 0}
            
            for class_est in preferences.CLASSES:
                
                self.conf_mat['event_based']['voting'][class_gt][class_est] = 0
                self.conf_mat['event_based']['emp_probs'][class_gt][class_est] = 0
                
                self.conf_mat['trajectory_based']['prob'][class_gt][class_est] = []
                self.conf_mat['trajectory_based']['binary'][class_gt][class_est] = 0
             

    def set_train_dists(self, train_fnames):
        
        # initialize empty histograms
        # since histogram is accumulated as below, it needs to be initialized 
        # at every training           
        for c in preferences.CLASSES:
            self.train_histograms[c] = {}
            self.train_pdfs[c] = {}
            for o in preferences.OBSERVABLES:
                self.train_histograms[c][o] = generic_tools.initialize_histogram(o)
                                
        # compute histograms for each class (using training set)
        for c in preferences.CLASSES:   
            for train_fname in train_fnames[c]:
                
                data = np.load(train_fname)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                
                for o in preferences.OBSERVABLES:
                    self.train_histograms[c][o] += generic_tools.compute_histogram_1D(o, obs_data[o])
                    
        for c in preferences.CLASSES:
            for o in preferences.OBSERVABLES:
                self.train_pdfs[c][o] = generic_tools.compute_pdf(o, self.train_histograms[c][o])


    
    def set_test_dists(self, test_fnames):
        
        # initialize empty histograms
        # since one histogram/pdf  is computed for each element of test set
        # as below, it needs to be initialized at every testing
        for c in preferences.CLASSES:
            self.test_histograms[c], self.test_pdfs[c] = {}, {}
            for test_fname in test_fnames[c]:
                self.test_histograms[c][test_fname], self.test_pdfs[c][test_fname] = {}, {}
                for o in preferences.OBSERVABLES:
                    self.test_histograms[c][test_fname][o] = generic_tools.initialize_histogram(o)
                    self.test_pdfs[c][test_fname][o] = []
                
        # compute histograms for each class (using test set)
        for c in preferences.CLASSES:   
            for test_fname in test_fnames[c]:

                data = np.load(test_fname)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                
                for o in preferences.OBSERVABLES:
                    self.test_histograms[c][test_fname][o] = generic_tools.compute_histogram_1D(o, obs_data[o])

        for c in preferences.CLASSES:
            for test_fname in test_fnames[c]:
                for o in preferences.OBSERVABLES:
                    self.test_pdfs[c][test_fname][o] = generic_tools.compute_pdf(o, self.test_histograms[c][test_fname][o])

                


    def estimate(self, test_fnames, m):
        """
        
        Performance is evaluated in various ways.
        
        -----------------------------------------------------------------------

        """

        for class_gt in preferences.CLASSES:
            for test_fname in test_fnames[class_gt]:
             
                # the following involve an array for each observable
                test_hists = self.test_histograms[class_gt][test_fname]
                test_pdfs = self.test_pdfs[class_gt][test_fname]            
                     
                distances  = {} 

                for class_query in preferences.CLASSES:
                    
                    # the following involve an array for each observable
                    train_pdfs = self.train_pdfs[class_query]
                    train_hists = self.train_histograms[class_query]
                
                    distances[class_query] = \
                    global_tools.compute_distance(m, \
                                                  train_pdfs, train_hists, \
                                                  test_pdfs, test_hists)
                p_est = global_tools.dist2prob(distances)
                                            
                ###############################################################
                #
                # only trajectory-based applies
                #
                for class_est in preferences.CLASSES:
                    # class_est is not really the 'output decision'
                    self.conf_mat['trajectory_based']['prob'][class_gt][class_est].append(\
                                 p_est[class_est])
                
                p_max = max(p_est.items(), key=operator.itemgetter(1))[1] 
                c_out = max(p_est.items(), key=operator.itemgetter(1))[0] 
                self.conf_mat['trajectory_based']['binary'][class_gt][c_out] += 1

                p_gt = p_est[class_gt]
                confidence = 1 - (p_max - p_gt)
                self.conf_mat['trajectory_based']['confidence'][class_gt].append(confidence)
                  
                
    
    
    def cross_validate(self):
        
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        out_fname = 'results/'+ year +'_'+ month +'_'+ day +'_'+ hour +'_'+ minute + \
        '_global_indep' + \
        '_nepoch_'+ str(preferences.N_EPOCH) +\
        '_train_ratio_' + str(preferences.TRAIN_RATIO*100) + \
        '_hier_' + (preferences.HIERARCHICAL) + '.txt'
        

        for m in preferences.GLOBAL_MEASURES:
            self.init_conf_mat()        
            for epoch in range(preferences.N_EPOCH):
                
                train_fnames, test_fnames = generic_tools.shuffle_data_fnames(self.data_fnames)
                self.set_train_dists(train_fnames)
                self.set_test_dists(test_fnames)
                
                self.estimate(test_fnames, m)
                
            for class_gt in preferences.CLASSES:
                self.conf_mat['trajectory_based']['confidence'][class_gt] = \
                np.mean(self.conf_mat['trajectory_based']['confidence'][class_gt])
                for class_est in preferences.CLASSES:
                    self.conf_mat['trajectory_based']['prob'][class_gt][class_est] = \
                    np.mean(self.conf_mat['trajectory_based']['prob'][class_gt][class_est])
                    
            generic_tools.write_conf_mat_to_file(out_fname, \
                                                 'global_indep', \
                                                 self.conf_mat, \
                                                 alpha_val=[], \
                                                 filtering_val=[], \
                                                 measure_val=m)
                    
                

            

               

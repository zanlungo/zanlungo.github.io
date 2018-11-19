import numpy as np
from scipy import spatial, stats
#from model import tools
import global_tools

import operator

# if the constants/preferences are not refreshed, take a long way and reload
import generic_tools
from importlib import reload
reload(generic_tools)


import preferences
from importlib import reload
reload(preferences)



class GlobalEstimator():
    """
    Global estimator class for the estimator
    """
    def __init__(self):
        self.train_pdfs = {}
        self.train_histograms = {}
        
        self.test_pdfs = {}
        self.test_histograms = {}
        
        self.data_fnames = generic_tools.get_data_fnames('data/classes/')


    def set_train_dists(self, train_fnames):
        
        # initialize empty histograms
        # since histogram is accumulated as below, it need to be initialized 
        # at every training
        for o in preferences.OBSERVABLES:
            self.train_histograms[o], self.train_pdfs[o] = {}, {}
            for c in preferences.CLASSES:
                self.train_histograms[o][c] = generic_tools.initialize_histogram(o)
                
        # compute histograms for each class
        for c in preferences.CLASSES:   
            for file_path in train_fnames[c]:
                data = np.load(file_path)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                for o in preferences.OBSERVABLES:
                    self.train_histograms[o][c] += generic_tools.compute_histogram_1D(o, obs_data[o])
                    
        for o in preferences.OBSERVABLES:
            for c in preferences.CLASSES:
                self.train_pdfs[o][c] = generic_tools.compute_pdf(o, self.train_histograms[o][c])


    
    def set_test_dists(self, test_fnames):
        
        # initialize empty histograms
        # since histogram is accumulated as below, it need to be initialized 
        # at every testing
        for o in preferences.OBSERVABLES:
            self.test_histograms[o], self.test_pdfs[o] = {}, {}
            for c in preferences.CLASSES:
                self.test_histograms[o][c], self.test_pdfs[o][c] = {}, {}
                for test_fname in test_fnames[c]:
                    self.test_histograms[o][c][test_fname] = generic_tools.initialize_histogram(o)
                    self.test_pdfs[o][c][test_fname] = []
                
        # compute histograms for each class
        for c in preferences.CLASSES:   
            for test_fname in test_fnames[c]:
                
                self.test_histograms[o][c][test_fname] = generic_tools.initialize_histogram(o)
                
                data = np.load(test_fname)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                for o in preferences.OBSERVABLES:
                    self.test_histograms[o][c][test_fname] = generic_tools.compute_histogram_1D(o, obs_data[o])
                    
        for o in preferences.OBSERVABLES:
            for c in preferences.CLASSES:
                for test_fname in test_fnames[c]:
                    self.test_pdfs[o][c][test_fname] = generic_tools.compute_pdf(o, self.test_histograms[o][c][test_fname])

                


    def estimate(self, test_fnames):
        results = {}
        for class_gt in preferences.CLASSES:
            results[class_gt] = {}
            for m in preferences.GLOBAL_MEASURES: 
                results[class_gt][m] = {}
                for o in preferences.OBSERVABLES:
                    results[class_gt][m][o] = {'right': 0, 'wrong': 0}
                    
                        
                        
        confusion_matrix = {}
        distances = {}

        # print('-------------------------------')
        # print('\t Right \t Wrong \t Rate\n')
 
                
        for class_gt in preferences.CLASSES:
            for test_fname in test_fnames[class_gt]:
                for m in preferences.GLOBAL_MEASURES: 
                    distances[m] = {}   
                    for o in preferences.OBSERVABLES:
                        distances[m][o] = {} 
                    
                        # init confusion matrix
                        confusion_matrix[class_gt] = {}
                        for class_pred in preferences.CLASSES:
                            confusion_matrix[class_gt][class_pred] = 0
                        
                        test_hist = self.test_histograms[o][class_gt][test_fname]
                        test_pdf = self.test_pdfs[o][class_gt][test_fname]
                        
                            
                        
                        for class_query in preferences.CLASSES:
                            
                            train_pdf = self.train_pdfs[o][class_query]
                            train_hist = self.train_histograms[o][class_query]
                        
                            distances[m][o][class_query] = \
                            global_tools.compute_distance(m, train_pdf, train_hist, test_pdf, test_hist)
                            
                        class_est = min(distances[m][o].items(), key=operator.itemgetter(1))[0]
                    
                        if class_est == class_gt:
                            results[class_gt][m][o]['right'] += 1
                        else:
                            results[class_gt][m][o]['wrong'] += 1
                    
        return results
    
    
    def cross_validate_binary(self):
        
        right_ns = {}
        for c in preferences.CLASSES:
            right_ns[c] = {}
            for m in preferences.GLOBAL_MEASURES:
                right_ns[c][m] = {}
                for o in preferences.OBSERVABLES:
                    right_ns[c][m][o] = []
                
        for epoch in range(preferences.N_EPOCH):
            
            train_fnames, test_fnames = generic_tools.shuffle_data_fnames(self.data_fnames)
            self.set_train_dists(train_fnames)
            self.set_test_dists(test_fnames)
            
            # evaluate bayesian
            results = self.estimate(test_fnames)
            
            for c in preferences.CLASSES:
                for m in preferences.GLOBAL_MEASURES:
                    for o in preferences.OBSERVABLES:
                        right_ns[c][m][o] += [results[c][m][o]['right']]
        
        # report the performance after all epochs are finished
        for m in preferences.GLOBAL_MEASURES:
            print('-------------------------------')
            for o in preferences.OBSERVABLES:
                 print('\n' + m + ' with ', o) 
                 tot_perf, tot_samp = 0, 0
                 for c in preferences.CLASSES:
                     mean_succ = np.mean(right_ns[c][m][o]) / len(test_fnames[c])
                     sdt_succ = np.std(right_ns[c][m][o]) / len(test_fnames[c])
                     tot_perf = tot_perf + mean_succ * len(self.data_fnames[c])
                     tot_samp = tot_samp + len(self.data_fnames[c])
                     print('{}\t {:.2f}% ± {:.2f}%'.format(c, mean_succ * 100, sdt_succ * 100))            
                 tot_perf = tot_perf / tot_samp
                 print('Tot\t {:.2f}%'.format(tot_perf*100))  
    
         


            

               

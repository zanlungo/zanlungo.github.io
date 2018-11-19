import numpy as np
from scipy import spatial
# for kde with gaussian kernels
from scipy import stats
# for kde with scikit 
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
# for ma filtering
from scipy import ndimage

#import matplotlib.pyplot as 
from os import listdir
import operator
import math
import time

# see warning control 
# https://docs.python.org/3/library/warnings.html
import warnings
warnings.simplefilter("default")

# if the constants/preferences are not refreshed, take a long way and reload
import preferences
from importlib import reload
reload(preferences)

import generic_tools
from importlib import reload
reload(generic_tools)



class BayesianEstimator():
    """
    Bayes class for the estimator
    """
    def __init__(self):
        self.train_pdfs_ND = {}
        self.kernels = {}
        self.data_fnames = generic_tools.get_data_fnames('data/classes/')

    def init_conf_mat(self):
        
        # init confusion matrix
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

                

    def train(self, train_fnames):
                       
        train_histograms_ND = {}
        for c in preferences.CLASSES:
                train_histograms_ND[c] = generic_tools.initialize_histogram_ND()
                
        # compute histograms for each class
        for c in preferences.CLASSES:   
            for file_path in train_fnames[c]:
                data = np.load(file_path)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                
                temp, edges = generic_tools.compute_histogram_ND( obs_data )
                
                train_histograms_ND[c] += temp
                    
            self.train_pdfs_ND[c] = generic_tools.compute_pdf_ND(train_histograms_ND[c])

    def trainKDE(self, train_fnames):
        
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        out_fname_bw_ests = 'results/bw_est_stability/'+ year +'_'+ month +'_'+ day +'_'+ hour +'_'+ \
        minute + '_'+ 'KDE_BW_estimation_stability.txt'
                         
        for c in preferences.CLASSES: 
            self.kernels[c] = []
            values = []
            
            for file_path in train_fnames[c]:
                data = np.load(file_path)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                
                # len(data_A) and len(data_B) are the same
                for j in range(0, len(data_A)): 
                    # prepare data point
                    data_pt = []
                    for o in preferences.OBSERVABLES:
                        data_pt.append(obs_data[o][j])
                        
                    values.append(data_pt)
                

            # optimizing kernel bandwidth with sklearn grid search
            params = {'bandwidth': np.linspace(preferences.BW0, preferences.BWF, preferences.NBINS_BW)}
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=preferences.NCV_BW)
            
            # I recently upgraded scikit-learn version to 0.21.dev0.
            # The following line gives deprecation warning:
            # DeprecationWarning: The default of the `iid` parameter will change 
            # from True to False in version 0.22 and will be removed in 0.24. 
            # This will change numeric results when test-set sizes are unequal.
            # DeprecationWarning)
            grid.fit(np.array(values))
          
            bw = grid.best_estimator_.bandwidth
            
            with open(out_fname_bw_ests, "a") as myfile:
                myfile.write(('{}\t{}\n'.format(c, bw)))
                
            

            self.kernels[c] = KernelDensity(bandwidth = bw, \
                        kernel='gaussian', algorithm='ball_tree')
            self.kernels[c].fit(np.array(values))
            
#            # Alternatively, you may set the BW by
#            # optimizing kernel bandwidth with gaussian_kde from stats package
#            # But scikit-learn is said to be more reliable so I stick to the above
#            m = len(preferences.OBSERVABLES)
#            n = len(values)     
#            self.kernels[c] = stats.gaussian_kde(np.reshape(values, (m,n)))
           
    def trainMA(self, train_fnames, sizeMA):
        """
        Apply a moving average filter over the pdfs
        """
                       
        train_histograms_ND = {}
        for c in preferences.CLASSES:
                train_histograms_ND[c] = generic_tools.initialize_histogram_ND()
                
        # compute histograms for each class
        for c in preferences.CLASSES:   
            for file_path in train_fnames[c]:
                data = np.load(file_path)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                
                temp, edges = generic_tools.compute_histogram_ND( obs_data )
                
                train_histograms_ND[c] += temp
                    
            temp = generic_tools.compute_pdf_ND(train_histograms_ND[c])
            self.train_pdfs_ND[c] = ndimage.uniform_filter(temp, size=preferences.SIZE_MA)
            
    def compute_probabilities_ND_woKDE(self, bins, alpha):
        """
        Estimate the probability of being in each possible class 
        The estimation is based on the ND distribution 
        WITHOUT kernel density estimation
        Therefore, it is sparse and there are many zeros...
        """
        
        n_data_pts = len(bins)
        p_posts, p_prior, p_likes, p_conds = {}, {}, {}, {}
        
        for c in preferences.CLASSES:
            p_prior[c] = 1 / len(preferences.CLASSES)
            p_posts[c] = np.zeros((n_data_pts))
            p_posts[c][0] = p_prior[c]
            
        for j in range(1, n_data_pts-1):
            for c in preferences.CLASSES:

                # I actually do notneed to init it to 1
                # because i do not accumulate o's like the 1D case
                p_likes[c] = 1
            
                # the following decreases the dimension at each iteration
                p_temp = self.train_pdfs_ND[c]
                for k, o in enumerate(preferences.OBSERVABLES):
                    p_temp = p_temp[bins[j][k]]
                    
                # do not use "p_temp is not 0"
                # the results may be a float (0.0) or an int (0)
                # it turns out that 0 and 0.0 are different 
                if p_temp > 0: 
                    p_likes[c] *= p_temp
                
                # if all p_temp are 0, then p_likes stay as 1 and 
                # the post is the same as prior
                p_prior[c] = alpha * p_posts[c][0] + (1 - alpha) * p_posts[c][j-1]
                p_conds[c] = p_likes[c] * p_prior[c]  
                
            s = sum(p_conds.values())
            if (s > 0) and (s is not np.nan):
                for c in preferences.CLASSES:
                    try: 
                        p_posts[c][j] = p_conds[c] / s 
                    except RuntimeWarning:
                        print('Error: sum of probabilities is 0 {} {}'.format(p_conds[c] , s ))
            else:
                print(s)
                for c in preferences.CLASSES:
                    p_posts[c][j] = 1/len(preferences.CLASSES)

        return p_posts

    def compute_probabilities_ND_wKDE(self, N_observations, obs_data, alpha):
        """
        Estimate the probability of being in each possible class 
        The estimation is based on the ND distribution 
        WITH kernel density estimation
        Therefore, the space is more populated

        """
        
        p_posts, p_prior, p_likes, p_conds = {}, {}, {}, {}
        
        for c in preferences.CLASSES:
            p_prior[c] = 1 / len(preferences.CLASSES)
            p_posts[c] = np.zeros((N_observations))
            p_posts[c][0] = p_prior[c]
            
            
        for j in range(1, N_observations-1):
            
            # prepare data point
            data_pt = []
            for o in preferences.OBSERVABLES:
                data_pt.append(obs_data[o][j])
                
            for c in preferences.CLASSES:

                p_likes[c] = 1       
                
                #with sklearn
                p_temp = self.kernels[c].score_samples([data_pt]) 
                p_temp = math.exp(p_temp) # since the above returns log
                
#                # with gaussian_kde
#                p_temp = self.kernels[c].evaluate(data_pt)  
                  
                if p_temp > 0:
                    # actually p_temp is never 0, 
                    # since the kernels spread over the entire space
                     try: 
                         p_likes[c] *= p_temp
                     except RuntimeWarning:
                        print('Problem with estimation with KDE: sum of probabilities is 0  {} {}'.format(p_likes[c] , p_temp ))
       
                
                # if all p_temp are 0, then p_likes stay as 1 and 
                # the post is the same as prior
                p_prior[c] = alpha * p_posts[c][0] + (1 - alpha) * p_posts[c][j-1]
                p_conds[c] = p_likes[c] * p_prior[c]   
                
            s = sum(p_conds.values())
            if (s > 0) and (s is not np.nan):
                for c in preferences.CLASSES:
                    try: 
                        p_posts[c][j] = p_conds[c] / s 
                    except RuntimeWarning:
                       # print('error pconds[c] {} plikes[c] {} p_prior[c] {}s {}'.format(\
                       #       p_conds[c] , p_likes[c], p_prior[c], s ))
                        
                        p_posts[c][j] = 1/len(preferences.CLASSES)
            else:
                for c in preferences.CLASSES:
                    p_posts[c][j] = 1/len(preferences.CLASSES)
                

        return p_posts

    def estimate(self, alpha, filtering, test_fnames):
    
        for class_gt in preferences.CLASSES:
                
            for t, test_fname in enumerate(test_fnames[class_gt]):
                 
                data = np.load(test_fname)
                data_A, data_B = generic_tools.extract_individual_data(data)
                N_observations = len(data_A) # len(data_B) is the same
                obs_data = generic_tools.compute_observables(data_A, data_B)
                
                
                if filtering is 'none':   
                    bins = generic_tools.find_bins_ND(obs_data)
                    p_posts =  self.compute_probabilities_ND_woKDE(bins, alpha) 
                    
                elif filtering is 'KDE':
                    p_posts = self.compute_probabilities_ND_wKDE(N_observations, obs_data, alpha)
                    
                elif filtering is 'MA':
                    bins = generic_tools.find_bins_ND(obs_data)
                    p_posts =  self.compute_probabilities_ND_woKDE(bins, alpha)
                    
                else:
                    print('bayesian_model_dep Line 293: preferences.FILTERING status is undefined')
                
                ###############################################################
                #
                # event based
                #

                n_votes = {}
                for class_temp in preferences.CLASSES:
                    n_votes[class_temp] = 0
                
                for i in range(0, N_observations):
                    # get all instantaneous probabilities
                    p_inst = {} #instantaneous probabilities
                    for class_temp in preferences.CLASSES:
                        p_inst[class_temp] = p_posts[class_temp][i]
                    
                    # the votes goes to the class with highest prob
                    class_est = max(p_inst.items(), key=operator.itemgetter(1))[0]
                    n_votes[class_est] += 1
                  
                class_est_voting_winner = max(n_votes.items(), key=operator.itemgetter(1))[0]
                self.conf_mat['event_based']['voting'][class_gt][class_est_voting_winner] += 1
                
                # scale the votes to 1, such that they represent probabilities
                factor = 1.0/sum(n_votes.values())
                class_est_emp_probs = {k: v*factor for k, v in n_votes.items() }

                
                for class_est in preferences.CLASSES:
                    # class_est is not really the 'output decision'
                    # here I only keep the probability associated with every 
                    # possible outcome
                    self.conf_mat['event_based']['emp_probs'][class_gt][class_est] += \
                    class_est_emp_probs[class_est]
                    
                ###############################################################
                #
                # trajectory-based
                #
                p_mean = {}
                for class_est in preferences.CLASSES:
                    # class_est is not really the 'output decision'
                    self.conf_mat['trajectory_based']['prob'][class_gt][class_est].append(\
                                 np.mean(p_posts[class_est]))
                    
                    p_mean[class_est] = np.mean(p_posts[class_est])
                    
                
                p_max = max(p_mean.items(), key=operator.itemgetter(1))[1] 
                c_out = max(p_mean.items(), key=operator.itemgetter(1))[0] 
                self.conf_mat['trajectory_based']['binary'][class_gt][c_out] += 1

                p_gt = p_mean[class_gt]
                
                confidence = 1 - (p_max - p_gt)
                self.conf_mat['trajectory_based']['confidence'][class_gt].append(confidence)
                
                ###############################################################
                #
                # collectively, ie dumping all observations from each class in 
                # one set, as if it is one long trajectory 
                #
                temp_suc = n_votes[class_gt]
                temp_fail = 0
                for class_est in preferences.CLASSES:
                    if class_est is not class_gt:
                        temp_fail += n_votes[class_est]
                        
                self.conf_mat['collective']['binary'][class_gt]['n_suc'] += temp_suc
                self.conf_mat['collective']['binary'][class_gt]['n_fail'] += temp_fail
                
                ###############################################################
                #
                # collective + confidence
                # There is lots of overlap between event-based 
                #
                temp_cum_n_observations = N_observations
                temp_cum_confidence = 0
                temp_cum_confidence_sq = 0

                for i in range(0, N_observations):
                    # get all instantaneous probabilities
                    p_inst = {} #instantaneous probabilities
                    for class_temp in preferences.CLASSES:
                        p_inst[class_temp] = p_posts[class_temp][i]
                    
                    # clas_est is the the one with highest prob
                    class_est = max(p_inst.items(), key=operator.itemgetter(1))[0]
                    
                    # p_est is the highest probability (ie the probability of 
                    # class_est). So I use p_est to compute confidence at this 
                    # instant
                    p_est = max(p_inst.items(), key=operator.itemgetter(1))[1]
                    temp = 1 - (p_est - p_inst[class_gt])
                    temp_cum_confidence += temp
                    temp_cum_confidence_sq += (temp*temp)
                    
                self.conf_mat['collective']['confidence'][class_gt]['cum_n_observations'] += temp_cum_n_observations
                self.conf_mat['collective']['confidence'][class_gt]['cum_confidence'] += temp_cum_confidence
                self.conf_mat['collective']['confidence'][class_gt]['cum_confidence_sq'] += temp_cum_confidence_sq
                
    
    
    def cross_validate(self, filtering):
        
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        out_fname = 'results/'+ year +'_'+ month +'_'+ day +'_'+ hour +'_'+ minute + \
        '_bayesian_dep' + \
        '_filter_' +  str(filtering) + \
        '_nepoch_'+ str(preferences.N_EPOCH) + \
        '_train_ratio_' + str(preferences.TRAIN_RATIO*100) + \
        '_hier_' + (preferences.HIERARCHICAL) + '.txt'

        
        
        for alpha in preferences.ALPHAS:

            self.init_conf_mat()            
               
            for epoch in range(preferences.N_EPOCH):
                
                train_fnames, test_fnames = generic_tools.shuffle_data_fnames(self.data_fnames)
                
                if filtering is 'none':
                    self.train(train_fnames)
                elif filtering is 'KDE':
                    self.trainKDE(train_fnames)
                elif filtering is 'MA':
                    self.trainMA(train_fnames, preferences.SIZE_MA)
                else:
                    print('bayesian_model_dep Line 350: preferences.FILTERING status {} is undefined'.format(filtering))
                    return 0

                self.estimate(alpha, filtering, test_fnames)
                
                

            for class_gt in preferences.CLASSES:
                # scale emp_probs
                # this is not readable
                self.conf_mat['event_based']['emp_probs'][class_gt] = \
                {k: v / total \
                 for total in (sum((self.conf_mat['event_based']['emp_probs']\
                                    [class_gt]).values()),) \
                                    for k, v in (self.conf_mat['event_based']['emp_probs']\
                                                 [class_gt]).items()}
                                    
                self.conf_mat['trajectory_based']['confidence'][class_gt] = \
                    np.mean(self.conf_mat['trajectory_based']['confidence'][class_gt])
                    
                for class_est in preferences.CLASSES:
                    
                    self.conf_mat['trajectory_based']['prob'][class_gt][class_est] = \
                    np.mean(self.conf_mat['trajectory_based']['prob'][class_gt][class_est])
                    
                    
                    
            generic_tools.write_conf_mat_to_file(out_fname, \
                                                 'bayesian_dep', \
                                                 self.conf_mat, \
                                                 alpha_val=alpha, \
                                                 filtering_val=[], \
                                                 measure_val=[])
                
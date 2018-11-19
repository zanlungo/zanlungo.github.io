import numpy as np
from scipy import spatial, stats
import time

#import matplotlib.pyplot as 
import operator

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
        self.train_pdfs1D = {}
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


    def train(self, train_fnames):
        
        train_histograms1D = {}
        # initialize empty histograms
        for o in preferences.OBSERVABLES:
            train_histograms1D[o], self.train_pdfs1D[o] = {}, {}
            for c in preferences.CLASSES:
                train_histograms1D[o][c] = generic_tools.initialize_histogram(o)
                
        # compute histograms for each class
        for c in preferences.CLASSES:   
            for file_path in train_fnames[c]:
                data = np.load(file_path)
                data_A, data_B = generic_tools.extract_individual_data(data)
                obs_data = generic_tools.compute_observables(data_A, data_B)
                for o in preferences.OBSERVABLES:
                    train_histograms1D[o][c] += generic_tools.compute_histogram_1D(o, obs_data[o])
                    
        for o in preferences.OBSERVABLES:
            for c in preferences.CLASSES:
                self.train_pdfs1D[o][c] = generic_tools.compute_pdf(o, train_histograms1D[o][c])

                
    def compute_probabilities(self, bins, alpha):
        
        n_data = len(bins[preferences.OBSERVABLES[0]])
        p_posts, p_prior, p_likes, p_conds = {}, {}, {}, {}
        
        for c in preferences.CLASSES:
            p_prior[c] = 1 / len(preferences.CLASSES)
            p_posts[c] = np.zeros((n_data))
            p_posts[c][0] = p_prior[c]
            
        for j in range(1, n_data):
            
            for c in preferences.CLASSES:
                p_likes[c] = 1
                for o in preferences.OBSERVABLES:
                    
#                    if self.train_pdfs1D[o][c][bins[o][j]-1] > 0:
#                        p_likes[c] *= self.train_pdfs1D[o][c][bins[o][j]-1]
                    if bins[o][j] < len(self.train_pdfs1D[o][c]): #constants.NBINS_PARAM_TABLE[o]
                        p_likes[c] *= self.train_pdfs1D[o][c][bins[o][j]]

                p_prior[c] = alpha * p_posts[c][0] + (1 - alpha) * p_posts[c][j-1]
                p_conds[c] = p_likes[c] * p_prior[c]  
                
            s = sum(p_conds.values())
            # if an onservation does not appear inof the classes, they will all 
            # get a 0 probability so the scaling below will give nan.
            # To avoid nans, I assume equal posteriors, which makes sense
            if s >0:
                for c in preferences.CLASSES:
                    p_posts[c][j] = p_conds[c] / s 
            else:
                for c in preferences.CLASSES:
                    p_posts[c][j] = 1 / len(preferences.CLASSES)
                        
                
        return p_posts

    def estimate(self, alpha, test_fnames):
        
        """
        
        Performance is evaluated in various ways.
        
        -----------------------------------------------------------------------
            
        event-based: treats each point on trajectory as an event. For each event,
        we make an instantaneous decision (koibito, yujin, etc). For instance, we
        have post probabilities as follows:
            
            time = t [K    D    Y    Kz]
            time = 0 [0.45 0.20 0.10 0.25]
            time = 1 [0.20 0.10 0.45 0.25]
            time = 2 [0.45 0.20 0.10 0.25]
            time = 3 [0.20 0.45 0.10 0.25]
            time = 4 [0.45 0.20 0.10 0.25]
            time = 5 [0.25 0.20 0.10 0.45]
            time = 6 [0.45 0.20 0.10 0.25]
            time = 7 [0.20 0.45 0.10 0.25]
           
        Each vector involves (post) probabilities for koibito (K), doryo (D), yujin (Y), 
        kazoku (Kz), respectively.
           
        Then the instantaneous decisions will be:
                [K Y K D K Kz K D]
        -----------------------------------------------------------------------
        event-based + voting: picks the class with highest number of votes among
        all events. So eventually the dyad has a single label (discrete output)
        For the above case, the votes are as follows:
            K = 4
            D = 2
            Y = 1
            Kz= 1
            
        So the output will be K. If this decision is correct it will give a
        1, otherwise a 0. Actually in the confusion matrix, I also store the 
        exact mistakes (off-diagonal).
            
        -----------------------------------------------------------------------
    
        event-based + empirical probability: the instantaneous  decisions are 
        expressed as empirical pribabilities. 
        
        For instance, for the above example, the empirical probabilities are:
            [4/8 2/8 1/8 1/8]
        for koibito (K), doryo (D), yujin (Y), kazoku (Kz), respectively.
           
        I use a confusion matrix to see the off-diagonal.        
        -----------------------------------------------------------------------
        trajectory-based: treats a trajectory as a single entity. 
        
        See below for details of :
            trajectory-based + prob 
            trajectory-based + confidence 
        
        -----------------------------------------------------------------------
        trajectory-based + prob: returns the probabilies of each possible 
        outcome as an average of probabilies at each time instant. 
        
        For the above case, we compute cumulative probabilities as an average of 
        probabilies at each time instant as follows:
            
            K = mean([.45, .20, .45, .20, .45, .25, .45, .20]) = 0.33125
            D = mean([.20, .10, .20, .45, .20, .20, .20, .45]) = 0.25
            Y = mean([.10, .45, .10, .10, .10, .10, .10, .10]) = 0.14375
            Kz= mean([.25, .25, .25, .25, .25, .45, .25, .25]) = 0.275
            
        -----------------------------------------------------------------------
        trajectory-based + binary: returns the class with highest probability as
        the output (decision) class.
        
        For the above case, the decision will be K.
            
            K = 0.33125
            D = 0.25
            Y = 0.14375
            Kz= 0.275
            
            argmax([K, D, Y, Kz]) = K
            
        -----------------------------------------------------------------------
        trajectory-based + confidence: returns a confidence metric which is defined 
        as below:
            conf = 100 - abs(p_max - p_gt)
        
        Here p_max is the highest probability (among the probabiities associated 
        with each possible outcome (ie class)). On the other hand, p_gt is the 
        probability that is associated with the gt class.
        
        For the above case, asuming thatthe gt class is D, conf will be:
            conf = 100 - abs(33.125 - 25)
                   = 91.875
            
        This value is 100 when the highest probability is associated with the gt class.
        When another class other than the gt class has a higher probability, it 
        gives the extent of the difference. 
        
        Values close to 100 indicate that there is a mistake but not that big.
        
        -----------------------------------------------------------------------
        collective: treats all observations from each gt class equally. Namely, 
        it boils down to four long trajectories for koibito, doryo, yujin and 
        kazoku.
        
        See below for details of :
            collective + confidence
            collective + binary

        -----------------------------------------------------------------------
        collective + confidence: I compute confidence at each single observation 
        point (ie trajectory point) 
        I do not store all these values. Instead, I store only the variables to 
        compute statistics. Namely:
            the number of observations
            the sum confidence values
            the sum of squares of confidence values
            
        -----------------------------------------------------------------------
        collective + binary: At each observation point, I make a binary decision
        and store the number of success and fails.
        
        The keys in the dictionary are:
            n_suc
            n_fail

        """
    
        for class_gt in preferences.CLASSES:
                
            for test_fname in test_fnames[class_gt]:
                 
                data = np.load(test_fname)
                data_A, data_B = generic_tools.extract_individual_data(data)
                N_observations = len(data_A) # len(data_B) is the same
                obs_data = generic_tools.compute_observables(data_A, data_B)
                                
                bins = {}
                for o in preferences.OBSERVABLES:
                    bins[o] = generic_tools.find_bins(o, obs_data[o])
                p_posts =  self.compute_probabilities(bins, alpha) 
                
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
                    # clas_est is the estimated class
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
                
    
    def cross_validate(self):
        
        # output file name
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        out_fname = 'results/'+ year +'_'+ month +'_'+ day +'_'+ hour +'_'+  minute + \
        '_bayesian_indep' + \
        '_nepoch_'+ str(preferences.N_EPOCH) +\
        '_train_ratio_' + str(preferences.TRAIN_RATIO*100) + \
        '_hier_' + (preferences.HIERARCHICAL) + '.txt'


        for alpha in preferences.ALPHAS:
            
            self.init_conf_mat()

            for epoch in range(preferences.N_EPOCH):
                
                train_fnames, test_fnames = generic_tools.shuffle_data_fnames(self.data_fnames)
                self.train(train_fnames)
                self.estimate(alpha, test_fnames)
                

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
                                                 'bayesian_indep', \
                                                 self.conf_mat, \
                                                 alpha_val=alpha, \
                                                 filtering_val=[], \
                                                 measure_val=[])
                
 
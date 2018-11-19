import numpy as np
from scipy import spatial, stats
from model import tools
#import matplotlib.pyplot as 
from os import listdir
import random
import operator
from model.constants import TRADUCTION_TABLE, PARAM_NAME_TABLE, PARAM_UNIT_TABLE, PLOT_PARAM_TABLE

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

def get_equal_datasets(data_path, classes):
    """
    Get the dataset for the given classes
    """
    datasets, sizes = {}, []
    for c in classes:
        class_path = data_path + c +'/'
        class_set = [class_path + f for f in listdir(class_path) if 'threshold' in f]
        sizes.append(len(class_set))
        datasets[c] = class_set
    min_size = min(sizes)
    for c in classes:
        shuffled_set = random.sample(datasets[c], len(datasets[c]))
        datasets[c] = shuffled_set[:min_size]
    return datasets

def shuffle_data_set(datasets, train_ratio):
    """
    Randomly partition the datasets into training and testing sets
    """
    train_sets, test_sets = {}, {}
    for c, dataset in datasets.items():
        n = len(dataset)
        n_train = round(train_ratio * n)
        shuffled_set = random.sample(dataset, n)
        train_sets[c] = shuffled_set[:n_train]
        test_sets[c] = shuffled_set[n_train:]
    
    return train_sets, test_sets

class BayesianEstimator():
    """
    Base class for the estimator
    """
    def __init__(self, cl, obs):
        self.cl = cl
        self.obs = obs
        self.pdfs = {}
        self.joint_pdfs = {}

    def train(self, train_sets):
        histograms, self.joint_pdfs, jaccard_dist = {}, {}, {}
        # initialize empty histograms
        for o in self.obs:
            histograms[o], self.pdfs[o] = {}, {}
            for c in self.cl:
                histograms[o][c] = tools.initialize_histogram(o)
        # compute histograms for each classes
        obs_data_cum = {}
        for c in self.cl:
            obs_data_cum[c] = {}
            for o in self.obs:
                     obs_data_cum[c][o] = []
            for file_path in train_sets[c]:
                data = np.load(file_path)
                data_A, data_B = tools.extract_individual_data(data)
                obs_data = tools.compute_observables(data_A, data_B)
                for o in self.obs:
                    histograms[o][c] += tools.compute_histogram(o, obs_data[o])
                    obs_data_cum[c][o].extend(obs_data[o])
                    
        for o in self.obs:
            for c in self.cl:
                self.pdfs[o][c] = tools.compute_pdf(o, histograms[o][c])
                
     
                
        for c in self.cl:
            self.joint_pdfs[c], jaccard_dist[c] = {}, {}
            for o1 in self.obs:
                self.joint_pdfs[c][o1], jaccard_dist[c][o1] = {}, {}
                for o2 in self.obs:
                    self.joint_pdfs[c][o1][o2] = tools.compute_joint_pdf(tools.compute_joint_histogram(o1, obs_data_cum[c][o1], o2, obs_data_cum[c][o2]))
                    joint_ent = tools.get_joint_ent(self.joint_pdfs[c][o1][o2], self.pdfs[o1][c], self.pdfs[o2][c])
                    mutual_inf = tools.get_mutual_inf(self.joint_pdfs[c][o1][o2], self.pdfs[o1][c], self.pdfs[o2][c])
                    # i should not need th follwoign wheck but all is nan 
                    if mutual_inf is not 0:
                        jaccard_dist[c][o1][o2] = (joint_ent - mutual_inf) / joint_ent
                    
        return jaccard_dist
               
    def get_KL_div(self,q, p):
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
    
    def get_KL_div_for_all_obs_at_once(self, q, p):
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
    
    def get_JS_div(self,q, p):
        """
        For discrete probability distributions P and Q, the Jensen-Shannon
        divergence from Q to P is defined to be
 
        D(P|Q) = 0.5sum{ P(i) log(M(i)/P(i)) } + 0.5sum{ Q(i) log(M(i)/Q(i)) }

        """
        # midpoint of p and q
        m = (p+q)/2

        div_symmetric = 0.5*self.get_KL_div( m, p) + 0.5*self.get_KL_div( m, q)
        
        return div_symmetric 
    
    def get_EMD(self, f1, f2):
       
        emd = []
        emd.append(0);
        
        if((np.sum(f1) - np.sum(f2)) > 2.220**(-10) ):
            print('sum(f1) = %5.5f  sum(f2) = %5.5f Make sure arrays are scaled' %(np.sum(f1),  np.sum(f2)) );
            return 0
                
        for i  in range(0, len(f1)):
            emd.append(f1[i] + emd[i] - f2[i])
        
        emd = np.sum(np.abs(emd))
        
        return emd
    
    def get_LL(self, query_hist, base_pdf):
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
   
    def get_dists(self, measure, testset_pdf, testset_hist, pdf):       
        """
        This is not nice
        """
        if measure is 'KLdiv':
            return self.get_KL_div(testset_pdf, pdf)
        elif measure is 'JSdiv':
            return self.get_JS_div(testset_pdf, pdf)
        elif measure is 'EMD':
            return self.get_EMD(testset_pdf, pdf)
        elif measure is 'LL':
            return self.get_LL(testset_hist, pdf)
        else:
            print('Unknown distance measure')
            return 0

    def get_dists_for_all_obs_at_once(self, measure, testset_pdfs, testset_hists, pdfs):       
        """
        This is not nice at all
        """
        if measure is 'KLdiv':
            return self.get_KL_div_for_all_obs_at_once(testset_pdfs, pdfs)
        elif measure is 'JSdiv':
            return self.get_JS_div_for_all_obs_at_once(testset_pdfs, pdfs)
        elif measure is 'EMD':
            return self.get_EMD_for_all_obs_at_once(testset_pdfs, pdfs)
        elif measure is 'LL':
            return self.get_LL_for_all_obs_at_once(testset_hists, pdfs)
        else:
            print('Unknown distance measure')
            return 0

    def eval_global(self, measures, test_sets):
        histograms, testset_pdfs = {}, {}
        
        dist_vals, conf_mats, results = {}, {}, {}
        for m in measures:
            dist_vals[m], conf_mats[m], results[m] = {}, {}, {}
            for c in self.cl:
                dist_vals[m][c], conf_mats[m][c], results[m][c] = {}, {}, {}
                for o in self.obs:
                    results[m][c][o] = {'right': 0, 'wrong': 0}
            
        for c in self.cl:
            histograms[c], testset_pdfs[c] = {}, {}
                                    
            for m in measures:  
                for c_pred in self.cl:
                    conf_mats[m][c][c_pred] = 0
                    
                for file_path in test_sets[c]:
                    histograms[c][file_path], testset_pdfs[c][file_path] = {}, {}
                    dist_vals[m][c][file_path] = {}
                      
                    data = np.load(file_path)
                    data_A, data_B = tools.extract_individual_data(data)
                    obs_data = tools.compute_observables(data_A, data_B)
    
                    for o in self.obs:
                        dist_vals[m][c][file_path][o] = {}
                            
                        histograms[c][file_path][o] = tools.initialize_histogram(o)
                        histograms[c][file_path][o] = tools.compute_histogram(o, obs_data[o])
                        testset_pdfs[c][file_path][o] = tools.compute_pdf(o, histograms[c][file_path][o])
                                           
                        for c_query in self.cl:
                            dist_vals[m][c][file_path][o][c_query] = self.get_dists(m, testset_pdfs[c][file_path][o], histograms[c][file_path][o], self.pdfs[o][c_query])
                                
                        c_pred = min(dist_vals[m][c][file_path][o].items(), key=operator.itemgetter(1))[0]
                        conf_mats[m][c][c_pred] += 1
                        if c_pred == c:
                            results[m][c][o]['right'] += 1
                        else:
                            results[m][c][o]['wrong'] += 1                                      
        return results
    
    
  

                
    def compute_probabilities(self, bins, alpha):
        n_data = len(bins[self.obs[0]])
        p_posts, p_prior, p_likes, p_conds = {}, {}, {}, {}
        for c in self.cl:
            p_prior[c] = 1 / len(self.cl)
            p_posts[c] = np.zeros((n_data))
            p_posts[c][0] = p_prior[c]
        for j in range(1, n_data):
            for c in self.cl:
                p_likes[c] = 1
                i = 0
                for o in self.obs:
                    if self.pdfs[o][c][bins[o][j]-1] != 0:
                        p_likes[c] *= self.pdfs[o][c][bins[o][j]-1]
                    # else:
                        # i += 1
                p_prior[c] = alpha * p_posts[c][0] + (1 - alpha) * p_posts[c][j-1]
                p_conds[c] = p_likes[c] * p_prior[c]          
            s = sum(p_conds.values())
            for c in self.cl:
                p_posts[c][j] = p_conds[c] / s 
        mean_ps = {}
        for c in self.cl:
            mean_ps[c] = np.mean(p_posts[c])
        return mean_ps

    def evaluate(self, alpha, test_sets):
        results = {}
        confusion_matrix = {}
        # print('-------------------------------')
        # print('\t Right \t Wrong \t Rate\n')
        t = 0
        for c in self.cl:
            results[c] = {'right': 0, 'wrong': 0}
            # init condusion matrix
            confusion_matrix[c] = {}
            for c_pred in self.cl:
                confusion_matrix[c][c_pred] = 0
            for file_path in test_sets[c]:
                data = np.load(file_path)
                data_A, data_B = tools.extract_individual_data(data)
                obs_data = tools.compute_observables(data_A, data_B)
                # obs_data = tools.shuffle_data(obs_data)
                bins = {}
                for o in self.obs:
                    bins[o] = tools.find_bins(o, obs_data[o])
                mean_p = self.compute_probabilities(bins, alpha)
                # t += i
                class_max = max(mean_p.items(), key=operator.itemgetter(1))[0]
                confusion_matrix[c][class_max] += 1
                if class_max == c:
                    results[c]['right'] += 1
                else:
                    results[c]['wrong'] += 1
                rate = results[c]['right'] / (results[c]['right'] + results[c]['wrong'])
            # print('{}\t {}\t {}\t {}'.format(c, results[c]['right'], results[c]['wrong'], rate))
        # tools.print_confusion_matrix(self.cl, confusion_matrix)
        # print(t)
        return results
    

    
    def evaluate_distance(self, alpha, test_sets):
        results = {}
        confusion_matrix = {}
        # print('-------------------------------')
        # print('\t Right \t Wrong \t Rate\n')
        t = 0
        for c in self.cl:
            results[c] = {'right': 0, 'wrong': 0}
            # init condusion matrix
            confusion_matrix[c] = {}
            for c_pred in self.cl:
                confusion_matrix[c][c_pred] = 0
            for file_path in test_sets[c]:
                data = np.load(file_path)
                pdfs, distances = {}, {}
                # initialize distances
                for c_pred in self.cl:
                    distances[c_pred] = 0
                data_A, data_B = tools.extract_individual_data(data)
                obs_data = tools.compute_observables(data_A, data_B)
                for o in self.obs:
                    pdfs[o] = tools.compute_pdf(o, tools.compute_histogram(o, obs_data[o]))
                    for c_pred in self.cl:
                        distances[c_pred] += stats.energy_distance(pdfs[o], self.pdfs[o][c_pred])
                # t += i
                class_max = min(distances.items(), key=operator.itemgetter(1))[0]
                confusion_matrix[c][class_max] += 1
                if class_max == c:
                    results[c]['right'] += 1
                else:
                    results[c]['wrong'] += 1
                rate = results[c]['right'] / (results[c]['right'] + results[c]['wrong'])
            # print('{}\t {}\t {}\t {}'.format(c, results[c]['right'], results[c]['wrong'], rate))
        # tools.print_confusion_matrix(self.cl, confusion_matrix)
        # print(t)
        return results

    
    def cross_validate(self, alphas, epoch, train_ratio, datasets):
        for a in alphas:
            right_ns = {}
            for c in self.cl:
                right_ns[c] = []
            for epoch in range(20):
                train_sets, test_sets = shuffle_data_set(datasets, train_ratio)
                self.train(train_sets=train_sets)
                
                # evaluate bayesian
                results = self.evaluate(alpha=a, test_sets=test_sets)
                
                for c in self.cl:
                    right_ns[c] += [results[c]['right']]
            print('-------------------------------')
            print('Bayesian')
            print('alpha = {}'.format(a))
            for c in self.cl:
                mean_succ = np.mean(right_ns[c]) / len(test_sets[c])
                sdt_succ = np.std(right_ns[c]) / len(test_sets[c])
                print('{}\t {:.2f}% ± {:.2f}%'.format(c, mean_succ * 100, sdt_succ * 100))
        
        
    def cross_validate_global(self, epoch, train_ratio, datasets, measures):
        
        right_ns, jaccard_dist = {}, []
        for m in measures:
            right_ns[m] = {}
            for c in self.cl:
                right_ns[m][c] = {}
                for o in self.obs:
                    right_ns[m][c][o] = []
            
                  
        for epoch in range(20):
            train_sets, test_sets = shuffle_data_set(datasets, train_ratio)
            jaccard_dist.append(self.train(train_sets=train_sets))
            results = self.eval_global(measures, test_sets = test_sets)
  
            for m in measures:
                for c in self.cl:
                    for o in self.obs: 
                        right_ns[m][c][o] += [results[m][c][o]['right']]
                      
        for m in measures:
            print('-------------------------------')
            for o in self.obs:
                 print(m + ' with ', o) 
                 tot_perf, tot_samp = 0, 0
                 for c in self.cl:
                     mean_succ = np.mean(right_ns[m][c][o]) / len(test_sets[c])
                     sdt_succ = np.std(right_ns[m][c][o]) / len(test_sets[c])
                     tot_perf = tot_perf + mean_succ * len(datasets[c])
                     tot_samp = tot_samp + len(datasets[c])
                     print('{}\t {:.2f}% ± {:.2f}%'.format(c, mean_succ * 100, sdt_succ * 100))            
                 tot_perf = tot_perf / tot_samp
                 print('Tot\t {:.2f}%'.format(tot_perf*100))   
        return jaccard_dist
                     
                
    def plot_pdf(self):
        plt.rcParams['grid.linestyle'] = '--'
        for o in self.obs:
            edges = tools.get_edges(o)
            for c in self.cl:
                plt.plot(edges, self.pdfs[o][c], label=TRADUCTION_TABLE[c], linewidth=3)
            plt.xlabel('{}({})'.format(PARAM_NAME_TABLE[o], PARAM_UNIT_TABLE[o]))
            plt.ylabel('p({})'.format(PARAM_NAME_TABLE[o]))
            plt.xlim(PLOT_PARAM_TABLE[o])
            plt.legend()
            plt.grid()
            plt.show()

    def plot_histogram(self, o, hist):
        plt.rcParams['grid.linestyle'] = '--'
        edges = tools.get_edges(o)
        plt.plot(edges, hist, linewidth=3)
        for c in self.cl:
            plt.plot(edges, self.pdfs[o][c], label=TRADUCTION_TABLE[c], linewidth=3)
        plt.xlabel('{}({})'.format(PARAM_NAME_TABLE[o], PARAM_UNIT_TABLE[o]))
        plt.ylabel('p({})'.format(PARAM_NAME_TABLE[o]))
        plt.xlim(PLOT_PARAM_TABLE[o])
        plt.grid()
        plt.show()


            

               

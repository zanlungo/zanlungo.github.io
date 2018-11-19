from model.bayesian_model import *
import numpy as np
import itertools

if __name__ == "__main__":

    classes = ['koibito', 'doryo', 'yujin', 'kazoku']
    observables = ['v_g', 'd', 'h_diff', 'v_diff']
    
    train_ratio = 0.3
    alphas = [0, 0.5, 1]
    epoch = 50

    for pair in itertools.combinations(classes, 2):
        bayesian_estimator = BayesianEstimator(cl=list(pair), obs=observables)
        datasets = get_datasets('data/classes/', list(pair))

        bayesian_estimator.cross_validate(alphas, epoch, train_ratio, datasets)
        print('=========================================')
import bayesian_model_indep
import bayesian_model_dep
import global_model_indep 
#import global_estimators_ND 

import numpy as np
import time

# if the constants/preferences are not refreshed, take a long way and reload
import preferences
from importlib import reload
reload(preferences)


if __name__ == "__main__":

    start_time = time.time()
    
    for method in preferences.METHODS:
        if method is 'bayesian_indep':
            # assuming independence
            bayesian_indep = bayesian_model_indep.BayesianEstimator()
            bayesian_indep.cross_validate()       
            
        elif(method is 'bayesian_dep'):
            # without assuming indepedence
             for filtering in preferences.FILTERING:
                 bayesian_dep = bayesian_model_dep.BayesianEstimator()
                 bayesian_dep.cross_validate(filtering)    
             
        elif(method is 'global_indep'):
            # alternative methods with assuming independence
            global_indep = global_model_indep.GlobalEstimator()
            global_indep.cross_validate()
         
    elapsed_time = time.time() - start_time
    print('\nTime elapsed  %2.2f sec' %elapsed_time)


    
TRAIN_RATIO = 0.3
N_EPOCH = 20

#'bayesian_indep', 'bayesian_dep', 'global_indep'
METHODS = ['bayesian_indep']

# options are 'stage1', 'stage2', 'off'
HIERARCHICAL = 'off'

# class options are 'koibito', 'doryo', 'yujin', 'kazoku', 'kazokuc', 'kazokua'
# kazokuc stand for kazoku with child (ie one member is younger than 15)
# kazokua stand for kazoku with all members adult (ie all members over 15)
# see split_kazoku for details
CLASSES_RAW = ['koibito', 'doryo', 'yujin', 'kazoku'] # used only in retrieving filenames at hier stage2

# if hierarchical is off, options are 'koibito', 'doryo', 'yujin', 'kazoku', 'kazokuc', 'kazokua'
# if hierarchical is on, it is either stage 1 or stage2
# In stage1, options are doryo and others
# In stage2, options are the ones except doryo
if HIERARCHICAL is 'off':
    CLASSES = ['koibito', 'doryo', 'yujin', 'kazoku']
elif HIERARCHICAL is 'stage1':
    CLASSES = ['doryo', 'others']
    OTHERS = [item for item in CLASSES_RAW if item not in CLASSES]
elif HIERARCHICAL is 'stage2':
    CLASSES = ['koibito', 'yujin', 'kazoku']
    

# 'v_g', 'v_diff', 'd', 'h_diff' 
OBSERVABLES = ['v_g', 'v_diff', 'd', 'h_diff' ]

# to be used in global methods
# 'KLdiv', 'JSdiv', 'EMD', 'LL'
GLOBAL_MEASURES = [ 'KLdiv', 'JSdiv', 'EMD', 'LL']

# to be used in updating the probabiilty in bayesian approach
ALPHAS = [0, 0.5, 1]

# to be used in DEP methods
# filtering options are 'none', 'KDE', 'MA'
# none: no filtering
# KDE: Kernel density estimation
# MA: Moving average
FILTERING =[ 'none', 'MA', 'KDE' ]

# to be used if filtering is set to 'MA'
SIZE_MA = 5 # support of moving average filter

# to be used if filtering is set to 'KDE'
# these parameters relate grid search for optimal kernel bandwidth
BW0 = 4 # lower bound of kernel bandwidth
BWF = 12# upper bound of kernel bandwidth
NBINS_BW = 100# number of bins between lower and upper bounds of kernel bandwidth
NCV_BW = 5# number of cross-vaidation

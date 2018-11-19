from model.constants import *
from model import tools
import numpy as np
import math
import glob

if __name__ == "__main__":
    
    MIN_NUMBER_SAMPLES = 30 # sample rate is 2samples/sec
    for file_path in glob.iglob('data/classes/**/*.dat', recursive=True):
        if 'threshold' not in file_path and 'day' in file_path:
            data = tools.load_data_scilab(file_path)
            data = tools.convert(data)
            threshold_data = tools.threshold(data)
            if np.shape(threshold_data[:,0])[0] > MIN_NUMBER_SAMPLES:
                threshold_file_path = '{}_threshold.dat'.format(file_path.rstrip('.dat'))
                print('Applying thresholds to {} and storing in {}'.format(file_path, threshold_file_path))
                with open(threshold_file_path, 'wb') as outfile:
                    np.save(outfile, threshold_data)
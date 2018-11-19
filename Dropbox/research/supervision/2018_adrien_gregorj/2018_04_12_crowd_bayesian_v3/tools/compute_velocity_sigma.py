import glob
from os import remove
from model import tools
import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    v_max = 0
    for file_path in glob.iglob('data/**/*.dat', recursive=True):
        if 'threshold' in file_path:
            data = np.load(file_path)
            data_A, data_B = tools.extract_individual_data(data)
            vxA, vyA = data_A[:, 5], data_A[:, 6]
            vxB, vyB = data_B[:, 5], data_B[:, 6]
            vxG, vyG = (vxA + vxB) / 2, (vyA + vyB) / 2
            vG = (vxG**2 + vyG**2)**.5
            if max(vG) > v_max:
                v_max = max(vG)
                
    print(v_max)


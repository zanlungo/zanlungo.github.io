import glob
from os import remove
from model import tools
import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    min_x, max_x = 0, 0
    min_y, max_y = 0, 0
    for file_path in glob.iglob('data/**/*.dat', recursive=True):
        if 'threshold' in file_path:
            data = np.load(file_path)
            data_A, data_B = tools.extract_individual_data(data)
            xA, yA = data_A[:, 2], data_A[:, 3]
            xB, yB = data_B[:, 2], data_B[:, 3]
            min_xA, min_xB = min(xA), min(xB)
            min_yA, min_yB = min(yA), min(yB)
            if min_xA < min_x:
                min_x = min_xA
            if min_xB < min_x:
                min_x = min_xB
            if min_yA < min_y:
                min_y = min_yA
            if min_yB < min_y:
                min_y = min_yB
            max_xA, max_xB = max(xA), max(xB)
            max_yA, max_yB = max(yA), max(yB)
            if max_xA > max_x:
                max_x = max_xA
            if max_xB > max_x:
                max_x = max_xB
            if max_yA > max_y:
                max_y = max_yA
            if max_yB > max_y:
                max_y = max_yB

    cell_size = 300
    ncell_x = math.ceil((max_x - min_x) / cell_size) + 1
    ncell_y = math.ceil((max_y - min_y) / cell_size) + 1

    occupancy_grid = np.zeros((ncell_x, ncell_y))

    for file_path in glob.iglob('data/**/*.dat', recursive=True):
        if 'threshold' in file_path:
            data = np.load(file_path)
            data_A, data_B = tools.extract_individual_data(data)
            xA, yA = data_A[:, 2], data_A[:, 3]
            xB, yB = data_B[:, 2], data_B[:, 3]
            for i in range(len(xA)):
                cell_coord_xA = math.ceil((xA[i] - min_x) / cell_size)
                cell_coord_yA = math.ceil((yA[i] - min_y) / cell_size)
                occupancy_grid[cell_coord_xA, cell_coord_yA] += 1
                cell_coord_xB = math.ceil((xB[i] - min_x) / cell_size)
                cell_coord_yB = math.ceil((yB[i] - min_y) / cell_size)
                occupancy_grid[cell_coord_xB, cell_coord_yB] += 1
    
    x,y = np.mgrid[slice(min_x, max_x + cell_size, cell_size), slice(min_y, max_y + cell_size, cell_size)]
    x /= 1000
    y /= 1000
    plt.pcolor(x, y, occupancy_grid)
    plt.show()


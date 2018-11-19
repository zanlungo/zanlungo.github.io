import glob
from os import remove

if __name__ == "__main__":
    for file_path in glob.iglob('data/**/*.dat', recursive=True):
        if 'threshold' in file_path:
            print('Removing {}'.format(file_path))
            remove(file_path)
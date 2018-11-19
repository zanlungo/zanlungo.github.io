from os import listdir

if __name__ == "__main__":
    classes = ['koibito', 'doryo', 'kazoku', 'yujin']
    n_files = {}
    for c in classes:
        class_path = 'data/classes/' + c
        n_files[c] = len([class_path + f for f in listdir(class_path) if 'threshold' in f])
    tot_files = sum(n_files.values())
    print(n_files)
    for c in classes:
        n_files[c] /= tot_files
    print(n_files)
    squared_p = {}
    for c in classes:
        squared_p[c] = n_files[c]**2
    print(squared_p)
    rnd_guess = sum(squared_p.values())
    print('Ramdom guess probability is {}'.format(rnd_guess))


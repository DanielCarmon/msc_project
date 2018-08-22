import sys
import os
import numpy as np
import pdb

project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project'
fname = sys.argv[1]
lines = open(fname).readlines()
paths = []
for line in lines:
    path = project_dir+'/'+line.split("'")[-2]
    paths.append(path)
print 'checking progress for jobs in ',fname
for path in paths:
    try:
        n_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    except:
        n_files=0
    n_ckpts = (n_files-1)/3
    path_suff = path.split('/')[-1]
    try: 
        n_train = len(np.load('train_data_scores'+path_suff+'.npy')[0])
    except:
        n_train = 0
    try:
        n_test = len(np.load('test_data_scores'+path_suff+'.npy')[0])
    except:
        n_test = 0
    print path_suff,'-->',n_ckpts,n_train,n_test


import glob
import sys
import time
import os
import numpy as np
import pdb
import signal
import sys

def signal_handler(sig, frame):
    print('Exiting on Ctrl+C')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

while True:
    project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project'
    fname = sys.argv[1]
    lines = open(fname).readlines()
    paths = []
    for line in lines:
        path = project_dir+'/'+line.split("'")[-2]
        paths.append(path)
    lines_to_print = ['checking progress for jobs in ',fname]
    for path in paths:
        try:
            list_of_files = glob.glob(path+'/*meta*')
            ns = [int(s.split('step_')[1].split('.')[0])for s in filter(lambda s: 'step' in s,list_of_files)]
            last_n = max(ns)/100
        except:
            last_n=0
        path_suff = path.split('/')[-1]
        try: 
            n_train = len(np.load('train_data_scores'+path_suff+'.npy')[0])
        except:
            n_train = 0
        try:
            n_test = len(np.load('test_data_scores'+path_suff+'.npy')[0])
        except:
            n_test = 0
        lines_to_print.append(path_suff+' --> '+str((last_n,n_train,n_test)))
    os.system('clear')
    for line in lines_to_print: print line
    time.sleep(0.1)

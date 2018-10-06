import glob
import sys
import time
import os
import numpy as np
import pdb
import signal
import sys
import re

def signal_handler(sig, frame):
    print('Exiting on Ctrl+C')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

while True:
    #pdb.set_trace()
    project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project'
    fname = sys.argv[1]
    lines = open(fname).readlines()
    paths = []
    for line in lines:
        model_name = '_'+'_'.join(re.split(',|:',line))
        path = project_dir+'/'+model_name
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
        ns = []
        for split in ['train','test','valid','minitrain','minitest']:
            try:
                n_split = len(np.load(split+'_data_scores'+path_suff+'.npy')[0])
                ns.append(n_split)
            except:
                ns.append(-1)
        lines_to_print.append(path_suff+' --> '+str((last_n,ns)))
    os.system('clear')
    for line in lines_to_print: print line
    time.sleep(0.1)

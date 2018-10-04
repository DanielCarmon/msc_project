import os
import os.path
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
os.environ["OMP_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
import glob
import tensorflow as tf
from sklearn import cluster
import traceback
import sys
from data_api import *
from model import *
from control.dcdb import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from tqdm import tqdm
from datetime import datetime
import pdb
import sys
import time
import traceback
import inspect
import pickle
from tensorflow.python import debug as tf_debug
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np

project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/'
logfile_path = project_dir+'/log_avg_models.txt'

def log_print(*msg):
    with open(logfile_path,'a+') as logfile:
        msg = [str(m) for m in msg]
        logfile.write(' '.join(msg))
        logfile.write('\n')
log_print('bzzzz')

def save(obj,fname):
    pickle_out = open(fname+'.pickle','wb+')
    pickle.dump(obj,pickle_out)
    pickle_out.close()

def as_bool(flag):
    if flag=='True':
        return True
    elif flag=='False':
        return False
    log_print('Error at parsing occured')
    pdb.set_trace()

config = tf.ConfigProto(allow_soft_placement=True)
sess_next = tf.Session()

d = 299
dataset_flag = 0
list_final_clusters = [100,98,512,102]
n_final_clusters = list_final_clusters[dataset_flag]
inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
embedder = InceptionEmbedder(inception_weight_path,new_layer_width=n_final_clusters)
startpoint = tf.placeholder(tf.float32,[None,299,299,3])
endpoint = embedder.embed(startpoint)

graph = tf.Graph()
saver = tf.train.Saver()

#sess_avg = tf.Session()
all_ckpt = range(1999-50,1999)
# Omitted code: Restore session1 and session2.
#pdb.set_trace()
for ckpt in all_ckpt:   
    print 'loading:',ckpt
    arch_name = '/_dataset_0_lr_1e-6_10_classes_tg_init++_em_3_iters'
    ckpt_path = project_dir+arch_name+'/step_{}.ckpt'.format(ckpt*100)
    saver.restore(sess_next,ckpt_path)
    print sess_next.run(embedder.params[1])[0]
    
# Optionally initialize session3.

"""
all_vars = tf.trainable_variables()
values1 = session1.run(all_vars)
values2 = session2.run(all_vars)

all_assign = []
for var, val1, val2 in zip(all_vars, values1, values2):
  all_assign.append(tf.assign(var, (val1 + val2)/ 2))

session3.run(all_assign)
"""
# Do whatever you want with session 3.

import os
import os.path
import tensorflow as tf
from sklearn import cluster
import traceback
import sys
from control import dcdb
from data_api import *
from model import *
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from tqdm import tqdm
from datetime import datetime
import pdb
import inspect
import pickle
from tensorflow.python import debug as tf_debug
from sklearn.metrics import normalized_mutual_info_score as skl_nmi
from utils import *

project_dir = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project"
logfile_path = project_dir+'/log_test.txt'

remote_run = True

def log_print(*msg):
    global remote_run
    if remote_run:
        with open(logfile_path,'a+') as logfile:
            msg = [str(m) for m in msg]
            logfile.write(' '.join(msg))
            logfile.write('\n')
    else:
        print [str(m) for m in msg]

log_print('entered restore_and_test.py')

argv = sys.argv
arg_dict = my_parser(argv)
log_print(now(),': using options:',arg_dict)
inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
dataset_flag = arg_dict['dataset']
use_permcovar = as_bool(arg_dict['permcovar'])
eval_split = int(arg_dict['eval_split'])
mini = eval_split>2
name = arg_dict['name']
log_print(name+': '+'Loading data... ')
data = get_data(eval_split,dataset_flag)
log_print(name+': '+'finished')
split_name = ['train','test','valid','minitrain','minitest'][eval_split]
fname_prefix = split_name+'_data_scores'
gpu = arg_dict['gpu']
if gpu>=0: os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto(allow_soft_placement=True)
log_print(name+': '+'Starting TF Session')
sess = tf.InteractiveSession(config=config)
d = 299
list_final_clusters = [100,98,512,102]
n_final_clusters = list_final_clusters[dataset_flag]
if mini: n_final_clusters = n_final_clusters/2
embedder = InceptionEmbedder(inception_weight_path,new_layer_width=n_final_clusters)
startpoint = tf.placeholder(tf.float32,[None,299,299,3])
endpoint = embedder.embed(startpoint,is_training=False)

if use_permcovar:
    embedder_pointwise = embedder
    embedder = PermCovarEmbedder1(n_final_clusters)
    n = data[0].shape[0]
    shape = [n,n_final_clusters]
    permcovar_startpoint = tf.placeholder(tf.float32,shape=shape) 
    permcovar_endpoint = embedder.embed(permcovar_startpoint)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

KMeans = cluster.KMeans
def test(test_data,use_permcovar=False): 
    tic = dcdb.now()
    global startpoint,endpoint
    log_print(name+': '+'begin test')
    test_xs,test_ys_membership = test_data
    n_test = test_xs.shape[0]
    n_clusters = test_ys_membership.shape[1]
    # 1) embed batch by batch
    def get_batch_iter(arr,bsz):
        i = 0
        n = arr.shape[0]
        while bsz*i<n:
            yield arr[bsz*i:bsz*(i+1)]
            i+=1

    def get_embedding(xs_batch,startpoint,endpoint):
        feed_dict = {startpoint:xs_batch}
        return sess.run(endpoint,feed_dict=feed_dict)
    output_dim = int(endpoint.shape[1].__str__()) # width of last layer in embedder
    
    i=0
    n_gpu_can_handle = 500
    batch_iter = get_batch_iter(test_xs,n_gpu_can_handle)
    embedding_list = []
    while True:
        try:
            xs_batch = batch_iter.next()       
            log_print(name+': '+'embedding batch ',i)
            embedded_xs_batch = get_embedding(xs_batch,startpoint,endpoint)
            embedding_list.append(embedded_xs_batch)
            i+=1
        except:
            log_print(name+': '+'finished inception embedding')
            np_embeddings = np.concatenate(embedding_list)
            break
    if use_permcovar:
        global permcovar_startpoint,permcovar_endpoint
        log_print(name+': '+'before ds module:',np_embeddings)
        feed_dict = {permcovar_startpoint: np_embeddings}
        np_embeddings = sess.run(permcovar_endpoint,feed_dict=feed_dict)
        log_print(name+': '+'after ds module:',np_embeddings)
    # 2) cluster
    log_print(name+': '+'clustering ',np_embeddings.shape[0],'vectors to ',n_clusters,'clusters...')
    km = KMeans(n_clusters=n_clusters).fit(np_embeddings)
    log_print(name+': '+'finished clustering')
    labels = km.labels_
    #labels_normalized = km_normalized.labels_
    nmi_score = skl_nmi(labels, np.argmax(test_ys_membership, 1))
    #nmi_score_normalized = skl_nmi(labels_normalized, np.argmax(ys_membership, 1))
    #scores = [nmi_score,nmi_score_normalized]
    result = nmi_score
    toc = dcdb.now()
    log_print(name+': '+'elapsed: {}'.format(toc-tic))
    return result
    
N = 6000
if mini: N = 6000
default_range_checkpoints = range(N) # might want to restore and test only a suffix of this
i_log = 100 # logging interval
results = []
cp_file_name = project_dir+'/'+fname_prefix+'{}.npy'.format(name) 
DIR = project_dir+'/{}'.format(name)
try: # load previous results, see what checkpoint was last restored and tested
    to_append = np.load(cp_file_name)[0]
    n_tests_already_made = len(to_append)
    range_checkpoints = default_range_checkpoints[n_tests_already_made:]
    if n_tests_already_made == N:
        log_print('{} evaluation completed'.format(name))
    else:
        log_print('restoring checkpoints in range',range_checkpoints[0],'to',range_checkpoints[-1])
except:
    log_print('no previous evaluations found. Evaluating from ckpt 0.')
    range_checkpoints = default_range_checkpoints
    to_append = []
for i in range_checkpoints:
    time.sleep(30) #TODO: del this
    log_print('testing for {}, checkpoint #{}. data split:{}'.format(name,i,str(eval_split)))
    ckpt_path = project_dir+'/'+name+'/step_{}'.format(i_log*i)+'.ckpt'
    if use_permcovar: log_print('WARNING: using permcovar' )
    #ckpt_path = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3/model.ckpt-157585' # for debug.
    try:
        #pdb.set_trace()
        with tf.device('/cpu:0'):
            saver.restore(sess,ckpt_path)
    except:
        log_print(name+': restore from '+ckpt_path+' failed. exiting program')
        exit(0)

    #save_var_dict(tf.global_variables(),'new_eval_bn_vars')
    #pdb.set_trace()
    result = test(data,use_permcovar)
    results.append(result)
    np.save(cp_file_name,[to_append+results,name]) # append new results by copying prev and rewriting 
    log_print(name+': '+'checkpoint result:',result)
log_print('*'*50)
log_print('results for {}:'.format(name),results)
log_print('*'*50)

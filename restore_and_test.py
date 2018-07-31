import os
import os.path
import tensorflow as tf
from sklearn import cluster
import traceback
import sys
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
from sklearn.metrics import normalized_mutual_info_score as nmi

def as_bool(s):
    if s=='False': return False
    if s=='True': return True
def my_parser(argv):
    ret = {}
    n = len(argv)
    for i in range(n):
        if argv[i][:2]=="--": # is flag
            val = argv[i+1]
            try:
                val = int(val)
            except:
                try:
                    val = float(val)
                except:
                    val = argv[i+1]
            ret[argv[i][2:]]=val
    return ret

argv = sys.argv
print 'tf version:',tf.__version__
print 'tf file:',tf.__file__
print 'python version:',sys.version_info 
arg_dict = my_parser(argv)
inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
project_dir = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project"

dataset_flag = arg_dict['dataset']
use_deepset = as_bool(arg_dict['deepset'])
test_last = bool(arg_dict['data_split'])
print 'Loading train data... '
data = get_data(test_last,dataset_flag)

if test_last:
    fname_prefix = 'test_data_scores'
else:
    fname_prefix = 'train_data_scores'
gpu = arg_dict['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto(allow_soft_placement=True)
print('Starting TF Session')
sess = tf.InteractiveSession(config=config)

d = 299
embed_dim = 1001
list_final_clusters = [100,98,512]
n_final_clusters = list_final_clusters[dataset_flag]
embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim,new_layer_width=n_final_clusters)
startpoint = tf.placeholder(tf.float32,[None,299,299,3])
endpoint = embedder.embed(startpoint)

if use_deepset:
    embedder_pointwise = embedder
    embedder = DeepSetEmbedder1(embed_dim)
    n = data[0].shape[0]
    shape = [n,embed_dim]
    deepset_startpoint = tf.placeholder(tf.float32,shape=shape) 
    deepset_endpoint = embedder.embed(deepset_startpoint)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

KMeans = cluster.KMeans
def test(test_data,use_deepset=False): 
    global startpoint,endpoint
    print 'begin test'
    test_xs,test_ys,test_ys_membership = test_data
    n_test = test_xs.shape[0]
    n_clusters = test_ys_membership.shape[1]
    # 1) embed batch by batch
    def get_embedding(xs_batch,startpoint,endpoint):
        feed_dict = {startpoint:xs_batch}
        return sess.run(endpoint,feed_dict=feed_dict)
    output_dim = int(endpoint.shape[1].__str__()) # width of last layer in embedder
    np_embeddings = np.zeros((0,output_dim))
    
    i=0
    n_batch = 400
    while n_batch*i<n_test:
        xs_batch = test_xs[n_batch*i:n_batch*(i+1)]       
        print 'embedding batch ',i
        embedded_xs_batch = get_embedding(xs_batch,startpoint,endpoint)
        np_embeddings = np.vstack((np_embeddings,embedded_xs_batch))
        i+=1
    if use_deepset:
        global deepset_startpoint,deepset_endpoint
        print 'before ds module:',np_embeddings
        feed_dict = {deepset_startpoint: np_embeddings}
        np_embeddings = sess.run(deepset_endpoint,feed_dict=feed_dict)
        print 'after ds module:',np_embeddings
    # 2) cluster
    km = KMeans(n_clusters=n_clusters).fit(np_embeddings)
    labels = km.labels_
    #labels_normalized = km_normalized.labels_
    nmi_score = nmi(labels, np.argmax(test_ys_membership, 1))
    #nmi_score_normalized = nmi(labels_normalized, np.argmax(ys_membership, 1))
    #scores = [nmi_score,nmi_score_normalized]
    result = nmi_score
    return result
pdb.set_trace()
N = 3000
default_range_checkpoints = range(N) # might want to restore and test only a suffix of this
i_log = 100 # logging interval

name = arg_dict['name']
results = []
cp_file_name = project_dir+'/'+fname_prefix+'{}.npy'.format(name)
DIR = project_dir+'/{}'.format(name)
try: # load previous results, see what checkpoint was last restored and tested
    to_append = np.load(cp_file_name)[0]
    n_tests_already_made = len(to_append)
    if n_tests_already_made == N:
        print '{} evaluation completed'.format(name)
    else:
        range_checkpoints = default_range_checkpoints[n_tests_already_made:]
        print 'restoring checkpoints in range',range_checkpoints[0],'to',range_checkpoints[-1]
except:
    print 'no previous evaluations found. Evaluating from ckpt 0.'
    range_checkpoints = default_range_checkpoints
    to_append = []
for i in range_checkpoints:
    print 'testing for {}, checkpoint #{}. test split?{}'.format(name,i,str(test_last))
    ckpt_path = project_dir+'/'+name+'/step_{}'.format(i_log*i)+'.ckpt'
    if use_deepset: print 'WARNING: using deepset' 
    #ckpt_path = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3/model.ckpt-157585' # for debug.
    saver.restore(sess,ckpt_path)
    result = test(data,use_deepset)
    results.append(result)
    np.save(cp_file_name,[to_append+results,name]) # append new results by copying prev and rewriting 
    print 'checkpoint result:',result
print '*'*50
print 'results for {}:'.format(name),results
print '*'*50

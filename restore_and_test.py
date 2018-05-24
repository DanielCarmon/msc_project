import os
import os.path
import tensorflow as tf
from sklearn import cluster
import traceback
import sys
from data_api import *
# from model import Model
from model import *
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from tqdm import tqdm
from datetime import datetime
import pdb
import sys
import traceback
import inspect
import pickle
from tensorflow.python import debug as tf_debug
from sklearn.metrics import normalized_mutual_info_score as nmi

inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
project_dir = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project"

dataset_flag = 1
use_deepset = False
test_last = True
print 'Loading train data... '
data = get_data(test_last,dataset_flag)
if test_last:
    fname_prefix = '101_to_200_scores'
else:
    fname_prefix = '1_to_100_scores'
sess = tf.InteractiveSession()

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

default_range_checkpoints = range(800) # might want to restore and test only a suffix of this
i_log = 100 # logging interval
# please make sure that dataset_flag is set correctly

#names = ['_lr_1e-5_tg_init++_em_1_iters','_lr_1e-6_tg_init++_em_1_iters','_lr_1e-7_tg_init++_em_1_iters']
names = ['_cars_lr_1e-5_tg_init++_em_1_iters']
for name in names:
    results = []
    cp_file_name = fname_prefix+'{}.npy'.format(name)
    DIR = project_dir+'/{}'.format(name)
    try: # load previous results, see what checkpoint was last restored and tested
        to_append = np.load(cp_file_name)[0]
        n_tests_already_made = len(to_append)
        range_checkpoints = default_range_checkpoints[n_tests_already_made:]
        print 'restoring checkpoints in range',range_checkpoints[0],'to',range_checkpoints[-1]
        #n_files = len([name_ for name_ in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name_))])
        #n_existing_ckpts = (n_files-1)/3 
    except:
        range_checkpoints = default_range_checkpoints
        to_append = []
    for i in range_checkpoints:
        print 'testing for {}, checkpoint #{}. test split?{}'.format(name,i,str(test_last))
        ckpt_path = project_dir+'/'+name+'/step_{}'.format(i_log*i)+'.ckpt'
        try:
            #ckpt_path = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3/model.ckpt-157585' # for debug.
            saver.restore(sess,ckpt_path)
            result = test(data,use_deepset)
            results.append(result)
            np.save(cp_file_name,[to_append+results,name]) # append new results by copying prev and rewriting 
            print 'checkpoint result:',result
        except:
            pass
    print '*'*50
    print 'results for {}:'.format(name),results
    print '*'*50

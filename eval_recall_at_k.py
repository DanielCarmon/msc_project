import tensorflow as tf
import numpy as np
import pdb
import os
import os.path
from sklearn import cluster
import traceback
import sys
from data_api import *
from model import *
from tqdm import tqdm
import pdb
import sys
import traceback
import pickle

def l2_normalize(x):
    # args:
    #   x- [n,d] np array
    norms = np.linalg.norm(x,axis=1)[:,np.newaxis]
    normalized = x/norms
    return normalized

inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/new_embed/inception-v3"
project_dir = "/specific/netapp5_2/gamir/carmonda/research/vision/new_embed"

dataset_flag = 0
use_permcovar = False
test_last = True
print 'Loading train data... '
data = get_data(test_last,dataset_flag)
if test_last:
    fname_prefix = '101_to_200_scores'
else:
    fname_prefix = '1_to_100_scores'
sess = tf.InteractiveSession()

d = 299
list_final_clusters = [100,98,512]
n_final_clusters = list_final_clusters[dataset_flag]
embedder = InceptionEmbedder(inception_weight_path,new_layer_width=n_final_clusters)
startpoint = tf.placeholder(tf.float32,[None,299,299,3])
endpoint = embedder.embed(startpoint)

if use_permcovar:
    embedder_pointwise = embedder
    embedder = PermCovarEmbedder1(n_final_clusters)
    n = data[0].shape[0]
    shape = [n,n_final_clusters]
    permcovar_startpoint = tf.placeholder(tf.float32,shape=shape) 
    permcovar_endpoint = embedder.embed(permcovar_startpoint)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

def get_dist_mat(a,b):
    norms_a = np.linalg.norm(a,axis=1)
    norms_b = np.linalg.norm(b,axis=1).T
    dps = np.matmul(a,b.T)
    return norms_a-2*dps+norms_b

def embed(x):
    global startpoint,endpoint
    def get_embedding(xs_batch,startpoint,endpoint):
        feed_dict = {startpoint:xs_batch}
        return sess.run(endpoint,feed_dict=feed_dict)
    output_dim = int(endpoint.shape[1].__str__()) # width of last layer in embedder
    np_embeddings = np.zeros((0,output_dim))
    
    i=0
    n_test = x.shape[0]
    n_batch = 400
    while n_batch*i<n_test:
        xs_batch = x[n_batch*i:n_batch*(i+1)]       
        print 'embedding batch ',i
        embedded_xs_batch = get_embedding(xs_batch,startpoint,endpoint)
        np_embeddings = np.vstack((np_embeddings,embedded_xs_batch))
        i+=1
    if use_permcovar:
        global permcovar_startpoint,permcovar_endpoint
        print 'before ds module:',np_embeddings
        feed_dict = {permcovar_startpoint: np_embeddings}
        np_embeddings = sess.run(permcovar_endpoint,feed_dict=feed_dict)
        print 'after ds module:',np_embeddings
    return np_embeddings

def eval_recalls(data,use_permcovar):
    test_xs,_,test_ys_membership = data
    ys = np.argmax(test_ys_membership,axis=1)
    n = test_xs.shape[0]
    embedded_data_path = project_dir+'/'+name+'_ckpt_{}.npy'.format(i_final)
    if os.path.isfile(embedded_data_path):
        print 'loading embedded data from',embedded_data_path
        embeddings = np.load(open(embedded_data_path,'r+'))
    else:
        print 'embedding data'
        embeddings = embed(test_xs)
        print 'saving to',embedded_data_path

        np.save(embedded_data_path,embeddings)
    dist_mat = get_dist_mat(embeddings,embeddings)
    ks = [1,2,4,8]
    counts = [0.]*len(ks)
    ret = []
    for i in range(n):
        y = ys[i]
        perm = np.argsort(dist_mat[i,:])
        for i_k in range(4):
            k = ks[i_k]
            knn_inds = perm[1:k+1]
            #pdb.set_trace()
            knn_labels = ys[knn_inds]
            counts[i_k]+=float(int(y in knn_labels))
    counts = [float(counts[i])/n for i in range(4)]
    return counts

i_log = 100 # logging interval
names = ['_lr_1e-6_tg_init++_em_1_iters']
#names = ['_lr_1e-5_tg_init++_em_1_iters','_lr_1e-6_tg_init++_em_1_iters','_lr_1e-7_tg_init++_em_1_iters']
for name in names:
    results = []
    DIR = project_dir+'/{}'.format(name)
    n_files = len([name_ for name_ in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name_))])
    n_existing_ckpts = (n_files-1)/3 
    i_final = n_existing_ckpts-1
    print 'evaluating recall@k score for {}, checkpoint #{}. test split?{}'.format(name,i_final,str(test_last))
    ckpt_path = project_dir+'/'+name+'/step_{}'.format(str(i_log*i_final))+'.ckpt'
    #ckpt_path = '/specific/netapp5_2/gamir/carmonda/research/vision/new_embed/inception-v3/model.ckpt-157585' # for debug.
    saver.restore(sess,ckpt_path)
    results = eval_recalls(data,use_permcovar)
    cp_file_name = project_dir+'/'+name+'_ckpt_{}_recall@k_score.npy'.format(i_final)
    np.save(cp_file_name,[results,name])
    print '*'*50
    print 'results for {}:'.format(name),results
    print '*'*50

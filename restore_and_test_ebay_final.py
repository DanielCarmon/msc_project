import os
import os.path
from control.dcdb import *
import sys
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
from control.dcdb import *

def as_bool(s):
    if s=='False': return False
    if s=='True': return True
def my_parser(argv):
    ret = {}
    n = len(argv)
    for i in range(n):
        if argv[i][:2]=="--":   # is flag
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
arg_dict = my_parser(argv)
inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
project_dir = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project"
dataset_flag = 2
use_permcovar = as_bool(arg_dict['permcovar'])
test_last = bool(arg_dict['data_split'])

if test_last:
    fname_prefix = 'ebay_total_test_data_scores'
else:
    fname_prefix = 'ebay_total_train_data_scores'
split = 'test' if test_last else 'train'
n_split = 59551 if split == 'train' else 60502
img_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/stanford_products/'+split+'/'
name = arg_dict['name']
# load proper weights: latest available
weight_dir = project_dir+'/{}'.format(name)
# get latest weight ckpt:
i = len(os.listdir(weight_dir))
i_log = 100
#i_ckpt = i_log*((i-1)/3-1)
i_ckpt = int(input('please specify ckpt to load\n'))
embedding_name = name+'_ckpt_'+str(i_ckpt)+'_embeddings.npy'
print i_ckpt
if arg_dict['embed']=='True': # we use gpu machine to embed the data to a [N,512] array
    gpu = arg_dict['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    config = tf.ConfigProto(allow_soft_placement=True)
    print('Starting TF Session')
    sess = tf.InteractiveSession(config=config)

    d = 299
    embedder = InceptionEmbedder(inception_weight_path,new_layer_width=512)
    startpoint = tf.placeholder(tf.float32,[None,d,d,3])
    endpoint = embedder.embed(startpoint)

    if use_permcovar:
        embedder_pointwise = embedder
        embedder = PermCovarEmbedder1(embed_dim)
        n = data[0].shape[0]
        shape = [n,embed_dim]
        permcovar_startpoint = tf.placeholder(tf.float32,shape=shape) 
        permcovar_endpoint = embedder.embed(permcovar_startpoint)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    print 'embedding ebay products with {}, checkpoint #{}. test split?{}'.format(name,i_ckpt,str(test_last))
    ckpt_path = weight_dir+'/step_{}'.format(i_ckpt)+'.ckpt'
    if use_permcovar: print 'WARNING: using permcovar' 
    #ckpt_path = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3/model.ckpt-157585' # for debug.
    saver.restore(sess,ckpt_path)

    # embed data split nad save to whatever.npy
    def get_embedding(xs_batch,startpoint,endpoint):
        feed_dict = {startpoint:xs_batch}
        return sess.run(endpoint,feed_dict=feed_dict)

    def get_batch_iter(arr,bsz):
        i = 0
        n = arr.shape[0]
        while i*bsz<n:
            yield arr[i*bsz:(i+1)*bsz]
            i+=1

    output_dim = int(endpoint.shape[1].__str__()) # width of last layer in embedder

    embedding_list = []
    n_gpu_can_handle = 500 # tested on 12GiB gpu
    shape = (n_split,299,299,3)
    imgs = np.memmap(img_dir+'memmap',dtype='float32',mode='r+',shape=shape)
    batch_iter = get_batch_iter(imgs,n_gpu_can_handle)
    done = 0

    while True:
        try:
            tic()
            batch = batch_iter.next()
            tmp_embedding = get_embedding(batch,startpoint,endpoint) 
            embedding_list.append(tmp_embedding)
            done+=tmp_embedding.shape[0]
            print 'embedded',done,'/',n_split,'({}%)'.format(100*float(done)/n_split)
            toc()
        except StopIteration:
            break
    np_embeddings = np.concatenate(embedding_list)
    try:
        np.save(embedding_name,np_embeddings)
    except:
        traceback.print_exc()
        pdb.set_trace()
    if use_permcovar:
        global permcovar_startpoint,permcovar_endpoint
        print 'before ds module:',np_embeddings
        feed_dict = {permcovar_startpoint: np_embeddings}
        np_embeddings = sess.run(permcovar_endpoint,feed_dict=feed_dict)
        print 'after ds module:',np_embeddings

else: # we use cpu to cluster embedding matrix
    print 'clustering ebay products with {}, checkpoint #{}. test split?{}'.format(name,i_ckpt,str(test_last))
    np_embeddings = np.load(embedding_name)
    print 'finished loading embeddings'
    class_szs = pickle.load(open(img_dir+'lengths.pickle'))
    membership_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_membership = block_diag(*membership_islands) # membership matrix
    n_clusters = ys_membership.shape[1]
    print name+': clustering',np_embeddings.shape[0],'vectors to',n_clusters,'clusters'
    tic()
    km = cluster.KMeans(n_clusters=n_clusters,verbose=1).fit(np_embeddings)
    toc()
    labels = km.labels_
    #labels_normalized = km_normalized.labels_
    nmi_score = nmi(labels, np.argmax(ys_membership, 1))
    #nmi_score_normalized = nmi(labels_normalized, np.argmax(ys_membership, 1))
    #scores = [nmi_score,nmi_score_normalized]
    result = nmi_score

    print '*'*50
    print 'checkpoint result:',result
    print '*'*50

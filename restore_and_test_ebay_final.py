import os
import os.path
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
from dcdb import *

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
use_deepset = as_bool(arg_dict['deepset'])
test_last = bool(arg_dict['data_split'])

if test_last:
    fname_prefix = 'ebay_total_test_data_scores'
else:
    fname_prefix = 'ebay_total_train_data_scores'
split = 'test' if test_last else 'train'
img_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/stanford_products/'+split+'/'

if arg_dict['embed']: # we use gpu machine to embed the data to a [N,512] array
    gpu = arg_dict['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    config = tf.ConfigProto(allow_soft_placement=True)
    print('Starting TF Session')
    sess = tf.InteractiveSession(config=config)

    d = 299
    embed_dim = 1001
    embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim,new_layer_width=512)
    startpoint = tf.placeholder(tf.float32,[None,d,d,3])
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

    # load proper weights: latest available
    name = arg_dict['name']
    weight_dir = project_dir+'/{}'.format(name)
    # get latest weight ckpt:
    i=0
    for filename in os.listdir(weight_dir):
        i+=1
        print filename
    i_log = 100
    i_relevant = i_log*((i-1)/3-1)
    print i_relevant
    print 'testing for {}, checkpoint #{}. test split?{}'.format(name,i_relevant,str(test_last))
    ckpt_path = weight_dir+'/step_{}'.format(i_relevant)+'.ckpt'
    if use_deepset: print 'WARNING: using deepset' 
    #ckpt_path = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3/model.ckpt-157585' # for debug.
    saver.restore(sess,ckpt_path)

    # embed data split nad save to whatever.npy
    def get_embedding(xs_batch,startpoint,endpoint):
        feed_dict = {startpoint:xs_batch}
        return sess.run(endpoint,feed_dict=feed_dict)

    output_dim = int(endpoint.shape[1].__str__()) # width of last layer in embedder
    np_embeddings = np.zeros((0,output_dim)) # init embedding matrix

    # load images batch by batch
    max_bytes = 5000
    tic()
    batch_bytes = 0
    img_batch = []
    for filename in list(filter(lambda x: x[:5]=='class',os.listdir(img_dir))):    
       print 'at',filename 
       class_imgs = np.load(img_dir+filename)
       class_imgs = class_imgs[:class_imgs.shape[0]/2]
       img_batch.append(class_imgs)
       sz = sys.getsizeof(img_batch)
       batch_bytes+=sz
       #print memsz(img_batch),sz
       if batch_bytes > max_bytes:
           print 'embedding...'
           toc()
           tic()
           batch_array = np.concatenate(img_batch)
           tmp_embedding = get_embedding(batch_array,startpoint,endpoint) 
           np_embeddings = np.vstack((np_embeddings,tmp_embedding))
           img_batch = []
           batch_bytes=0
    toc()
    np.save(split+'_embeddings.npy',np_embeddings)
    pdb.set_trace()
    if use_deepset:
        global deepset_startpoint,deepset_endpoint
        print 'before ds module:',np_embeddings
        feed_dict = {deepset_startpoint: np_embeddings}
        np_embeddings = sess.run(deepset_endpoint,feed_dict=feed_dict)
        print 'after ds module:',np_embeddings

    #print 'Embedded data to:',path_to_embed_mat
else: # we use cpu to cluster embedding matrix
    np_embeddings = np.load(split+'_embeddings.npy')
    class_szs = pickle.load(open(img_dir+'lengths.pickle'))
    membership_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_membership = block_diag(*membership_islands) # membership matrix
    n_clusters = ys_membership.shape[1]
    print 'lets fit'
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

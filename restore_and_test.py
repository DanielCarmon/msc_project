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

# stuff which should be decided about:
use_deepset = False # not implemented yet
use_crop = False
name = 'debug_runs'
sess = tf.InteractiveSession()

d = 299
data_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds'
inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
embed_dim = 1001
embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim)

if use_deepset:
    embedder_pointwise = embedder
    embedder = DeepSetEmbedder1(embed_dim).compose(embedder_pointwise) # Under Construction!
# prepare test data
print 'Loading train data... '
pdb.set_trace()
first_100_data = load_specific_data(data_dir,range(1,3),use_crop=use_crop)
print 'Loading test data... '
last_100_data = load_specific_data(data_dir,range(2,4),use_crop=use_crop)
KMeans = cluster.KMeans
def test(test_data): 
    print 'begin test'
    test_xs,test_ys,test_ys_membership = test_data
    n_test = test_xs.shape[0]
    # 1) embed batch by batch
    def get_embedding(xs_batch,startpoint,endpoint):
        feed_dict = {startpoint:xs_batch}
        return sess.run(endpoint,feed_dict=feed_dict)
    np_embeddings = np.zeros((0,embed_dim))
    startpoint,endpoint = model.x,model.x_embed
    if arg_dict['deepset']: # need to batchwise apply pointwise embedder
        #startpoint = tf.placeholder(tf.float32,[None,299,299,3])
        startpoint = model.x
        #endpoint = embedder_pointwise.embed(startpoint)
        endpoint = embedder_pointwise.out #?
        pdb.set_trace()
        print 'meow'
    i=0
    n_batch = 400
    while n_batch*i<n_test:
        xs_batch = test_xs[n_batch*i:n_batch*(i+1)]       
        print 'embedding batch ',i
        embedded_xs_batch = get_embedding(xs_batch,startpoint,endpoint)
        np_embeddings = np.vstack((np_embeddings,embedded_xs_batch))
        i+=1
    np_embeddings_normalized = l2_normalize(np_embeddings)
    # 2) cluster
    km = KMeans(n_clusters=n_test_classes).fit(np_embeddings)
    #km_normalized = KMeans(n_clusters=n_test_classes).fit(np_embeddings_normalized)
    labels = km.labels_
    #labels_normalized = km_normalized.labels_
    nmi_score = nmi(labels, np.argmax(test_ys_membership, 1))
    #nmi_score_normalized = nmi(labels_normalized, np.argmax(ys_membership, 1))
    #scores = [nmi_score,nmi_score_normalized]
    results = nmi_score
    return results

test_scores = []
train_scores = []
#test_scores.append(test(last_100_data)) # check over 100 last classes
#train_scores.append(test(first_100_data)) # check over 100 first classes
print '*'*50
print 'test scores:',test_scores
print '*'*50

cp_file_name = 'test_scores{}.npy'.format(name)
np.save(cp_file_name,[test_scores,name]) 
#train_scores.append(test(first_100_data)) # check over 100 first classes
print '*'*50
print 'test scores:',test_scores
print '*'*50

cp_file_name = 'test_scores{}.npy'.format(name)
np.save(cp_file_name,[test_scores,name]) 

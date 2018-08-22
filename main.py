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
from dcdb import *
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

def save(obj,fname):
    pickle_out = open(fname+'.pickle','wb+')
    pickle.dump(obj,pickle_out)
    pickle_out.close()

def as_bool(flag):
    if flag=='True':
        return True
    elif flag=='False':
        return False
    print 'Error at parsing occured'
    pdb.set_trace()

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
    ret['deepset'] = as_bool(ret['deepset']) 
    #ret['use_tg'] = as_bool(ret['use_tg']) 
    ret['use_tg'] = True
    ret['init'] = 2
    ret['obj'] = 'L2'
    ret['cluster'] = 'em'
    ret['cluster_hp'] = 1e-2
    if not 'train_params' in ret.keys():
        ret['train_params'] = 'e2e'
    ret['restore_last'] = True
    print 'using options:',ret
    if ret['cluster'] == "em":
        ret['cluster'] = EMClusterer
    if ret['cluster'] == "kmeans":
        ret['cluster'] = GDKMeansClusterer2
    if not 'n_test_classes' in ret.keys():
        ret['n_test_classes'] = 100
    return ret

def linenum():
    """ Returns current line number """
    return inspect.currentframe().f_back.f_lineno

def get_tb():
    exc = sys.exc_info()
    return traceback.print_exception(*exc)

def trim(vec, digits=3):
    factor = 10 ** digits
    vec = np.round(vec * factor) / factor
    return vec


project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/'

data_params = None
k = None
embedder = None
clusterer = None
tf_clustering = None

def run(arg_dict):
    global embedder, clusterer, tf_clustering, data_params, k, sess
    d = 299
    k = 2
    if 'n_train_classes' in arg_dict.keys():
        k = arg_dict['n_train_classes']
    name = arg_dict['name']
    dataset_flag = arg_dict['dataset']
    print 'loading data...'
    init_train_data(dataset_flag)
    print 'done'
    n_gpu_can_handle = 100
    n_ = n_gpu_can_handle/k # points per cluster
    n = n_*k
    recompute_ys = dataset_flag==2 # only recompute for products dataset
    clst_module = arg_dict['cluster']
    hp = arg_dict['cluster_hp'] # hyperparam. bandwidth for em, step-size for km
    model_lr = arg_dict['model_lr']
    use_tg = arg_dict['use_tg']
    data_params = [n, d]
    inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
    #vgg_weight_path = '/specific/netapp5_2/gamir/carmonda/research/vision/vgg16_weights.npz'
    #weight_path = '/home/d/Desktop/uni/research/vgg16_weights.npz'
    #embed_dim = 128
    # embedder = Vgg16Embedder(vgg_weight_path,sess=sess,embed_dim=embed_dim)
    embed_dim = 1001
    init = arg_dict['init']
    list_final_clusters = [100,98,512,102]
    n_final_clusters = list_final_clusters[dataset_flag] # num of clusters in dataset
    embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim,new_layer_width=n_final_clusters)
   
    if arg_dict['deepset']:
        embedder_pointwise = embedder
        embedder = DeepSetEmbedder1(n_final_clusters).compose(embedder_pointwise) # Under Construction!
    clusterer = clst_module([n, embed_dim], k, hp, n_iters=arg_dict['n_iters'],init=init)

    print 'building model object'
    log_grads = False
    obj = arg_dict['obj']
    model = Model(data_params, embedder, clusterer, model_lr, is_img=True,sess=sess,for_training=False,regularize=False, use_tg=use_tg,obj=obj,log_grads=log_grads)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
    n_offset = 0 # no. of previous checkpoints
    try: # restore last ckpt
        if arg_dict['restore_last']:
            ckpt_path = project_dir+name
            list_of_files = glob.glob(ckpt_path+'/*meta*')
            ns = [int(s.split('step_')[1].split('.')[0])for s in filter(lambda s: 'step' in s,list_of_files)]
            last_n = max(ns)
            ckpt_path = ckpt_path+'/step_'+str(last_n)+'.ckpt'
            n_offset = last_n+1
            print 'Restoring parameters from',ckpt_path
            saver.restore(sess,ckpt_path)
            nmi_score_history_prefix = np.load(project_dir+'train_nmis{}.npy'.format(name))
            loss_history_prefix = np.load(project_dir+'train_losses{}.npy'.format(name))
    except: 
        print 'no previous checkpoints found'
        nmi_score_history_prefix = []
        loss_history_prefix = []

    def train(model,hyparams,name):
        global test_scores_em,test_scores_km # global so it could be reached at debug pm mode
        test_scores = []
        train_scores = []
        n_steps,k,n_,i_log  = hyparams
        param_history = []
        loss_history = []
        nmi_score_history = []
        step = model.train_step

        for i in range(n_offset,n_steps): 
            xs, ys = get_train_batch(dataset_flag,k,n,recompute_ys=recompute_ys)
            feed_dict = {model.x: xs, model.y: ys}
            if (i%i_log==0): # case where i==0 is baseline
                print 'meow'
                nmi_2_save = list(nmi_score_history_prefix)+nmi_score_history
                np.save(project_dir+'train_nmis{}.npy'.format(name),np.array(nmi_2_save))
                l2_2_save = list(loss_history_prefix)+loss_history
                np.save(project_dir+'train_losses{}.npy'.format(name),np.array(l2_2_save))
                saver.save(sess,project_dir+"{}/step_{}.ckpt".format(name,i)) 
                print 'woem'
            try:
                print now(),': train iter',i,' for',name
                #pdb.set_trace()
                #activations_tensors = sess.run(embedder.activations_dict,feed_dict=feed_dict)
                #print 'before embed'
                #embed = sess.run(model.x_embed,feed_dict=feed_dict) # embeddding for debug. see if oom appears here.
                #print 'after embed'
                _,clustering_history,clustering_diffs,loss,grads = sess.run([step,clusterer.history_list, clusterer.diff_history,model.loss, model.grads], feed_dict=feed_dict)
            #_,activations,parameters,clustering_history,clustering_diffs = sess.run([step,embedder.activations_dict,embedder.param_dict,model.clusterer.history_list,clusterer.diff_history], feed_dict=feed_dict) 
                clustering = clustering_history[-1]
                # ys_pred = np.matmul(clustering,clustering.T)
                # ys_pred = [[int(elem) for elem in row] for row in ys_pred] 
                nmi_score = nmi(np.argmax(clustering, 1), np.argmax(ys, 1))
                print 'after: ',nmi_score
                nmi_score_history.append(nmi_score)
                loss_history.append(loss)
                print "clustring diffs:",clustering_diffs
            except:
                print 'error occured'
                exc =  sys.exc_info()
                traceback.print_exception(*exc)
                pdb.set_trace()
        print 'train_nmis:',nmi_score_history
        return nmi_score_history,test_scores

    print 'begin training'
    # end-to-end training:
    i_log = 100 
    n_train_iters = 4500
    hyparams = [n_train_iters*i_log,k,n_,i_log]
    test_scores_e2e = []
    test_scores_ll = []
    if arg_dict['train_params']!="last":
        try:
            train_nmis,test_scores_e2e = train(model,hyparams,name)
        except:
            print get_tb()
            exit()
            pdb.set_trace()
    else:
        print 'not training e2e'
        print 'starting last-layer training'
        if arg_dict['deepset']:
            filter_cond = lambda v: 'DeepSet' in str(v)
        else:
            filter_cond = lambda x: ("logits" in str(x)) and not ("aux" in str(x))
        deepset_params = filter(filter_cond,tf.global_variables())
        last_layer_params = filter(filter_cond,tf.global_variables())
        if not arg_dict['deepset']: last_layer_params.append(embedder.new_layer_w)
        model.train_step = model.optimizer.minimize(model.loss, var_list=last_layer_params) # freeze all other weights
        train_nmis,test_scores_ll = train(model,hyparams,name)
    #save_path = embedder.save_weights(sess)
    print 'end training' 
    return train_nmis

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
if __name__ == "__main__":
    print 'entered main:',datetime.now()
    argv = sys.argv
    arg_dict = my_parser(argv)
    gpu = arg_dict['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    #config = tf.ConfigProto(allow_soft_placement=True)
    print('Starting TF Session')
    #sess = tf.InteractiveSession(config=config)
    sess = tf.InteractiveSession()
    run(arg_dict)

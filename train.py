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
import pdb #line was here
import sys
import time
import traceback
import inspect
import pickle
from tensorflow.python import debug as tf_debug
from sklearn.metrics import normalized_mutual_info_score as skl_nmi
import numpy as np

project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/'
logfile_path = project_dir+'/log_train.txt'

def log_print(*msg):
    with open(logfile_path,'a+') as logfile:
        msg = [str(m) for m in msg]
        logfile.write(' '.join(msg))
        logfile.write('\n')
log_print(now(),': bzzzz')

data_params = None
k = None
embedder = None
clusterer = None

def save(obj,fname):
    pickle_out = open(fname+'.pickle','wb+')
    pickle.dump(obj,pickle_out)
    pickle_out.close()

def my_parser(argv):
    ret = {} 
    # default opts:
    ret['use_tg'] = True # aux gradients
    ret['obj'] = 'L2' # distance between pred and gt
    ret['cluster'] = 'em' # cluster inference module
    ret['cluster_hp'] = 1e-2 # bandwidth for em, step-size for km
    ret['params'] = 'e2e' # what params to train
    ret['init'] = 2 # init method for clusterer
    ret['restore_last'] = True # load params from last ckpt
    # override defaults:
    n = len(argv)
    for i in range(n):
        if argv[i][:2]=="--": # is flag
            key = argv[i][2:]
            val_raw = argv[i+1]
            try:
                val = eval(val_raw)
            except: # val should be string
                val = val_raw 
            ret[key]=val
    # format opts:        
    name = ret['name']
    ret['permcovar'] = bool(ret['permcovar']) 
    log_print(now(),': using options:',ret)
    if ret['cluster'] == "em":
        ret['cluster'] = EMClusterer
    if ret['cluster'] == "kmeans":
        ret['cluster'] = GDKMeansClusterer2
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



def run(arg_dict):
    global embedder, clusterer, data_params, k, sess # initialized as None. kept as global for access in debugger mode
    name = arg_dict['name']
    # dataset configs:
    dataset_flag = arg_dict['dataset']
    mini = arg_dict['split'] > 2 # train on miniset
    log_print(now(),': started loading data for',name,'...')
    init_train_data(dataset_flag,mini=mini,name=name)
    log_print(now(),': finished loading data for',name,'...')
    #pdb.set_trace()
    list_final_clusters = [100,98,512,102] # categories per dataset
    n_final_clusters = list_final_clusters[dataset_flag] # num of clusters in dataset
    if mini: # train on miniset for val
        n_final_clusters = n_final_clusters/2
    
    # batch configs:
    d = 299 # inception input size = [d,d]
    k = arg_dict['n_classes'] # clusters per batch
    n_gpu_can_handle = 100 # batch size
    n_ = n_gpu_can_handle/k # points per cluster
    n = n_*k
    data_params = [n, d]

    # data handle configs:
    recompute_ys = dataset_flag==2 # only recompute for products dataset
    n_ebay_batches = 50 # number of preprocessed samples for products dataset

    # model configs:
    clst_module = arg_dict['cluster'] # clusterer to use
    hp = arg_dict['cluster_hp'] # see arg_parser
    lr = arg_dict['lr'] # base learning-rate for inception
    use_tg = arg_dict['use_tg'] # use aux gradients
    init = arg_dict['init'] # init method for clusterer
    inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3" # params pre-trained on ImageNet
    embedder = InceptionEmbedder(inception_weight_path,new_layer_width=n_final_clusters) # embedding function  
    if arg_dict['permcovar']: # if using permcovar layers. still experimental
        embedder_pointwise = embedder
        embedder = PermCovarEmbedder1(n_final_clusters).compose(embedder_pointwise)
    clusterer = clst_module([n,n_final_clusters], k, hp, n_iters=arg_dict['n_iters'],init=init)
    log_print(now(),': building model for',name)
    log_grads = False
    obj = arg_dict['obj']
    model = Model(data_params, embedder, clusterer, lr, is_img=True,sess=sess,for_training=False,regularize=False, use_tg=use_tg,obj=obj,log_grads=log_grads) # compose clusterer on embedder and add loss function

    # ckpt configs:
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
            log_print(now(),': Restoring parameters from',ckpt_path)
            saver.restore(sess,ckpt_path)
            nmi_score_history_prefix = np.load(project_dir+'train_nmis{}.npy'.format(name))
            loss_history_prefix = np.load(project_dir+'train_losses{}.npy'.format(name))
    except: 
        log_print(now(),': no previous checkpoints found for',name)
        nmi_score_history_prefix = []
        loss_history_prefix = []

    def train(model,hyparams,name): # training procedure
        global test_scores_em,test_scores_km # global so it could be reached at debug pm mode
        test_scores = []
        train_scores = []
        n_steps,k,n_,i_log  = hyparams
        param_history = []
        loss_history = []
        nmi_score_history = []
        step = model.train_step # update step op
        
        current_batch = 0 # only relevant for ebay dataset
        for i in range(n_offset,n_steps): # main loop
            if (i%i_log==0): # save ckpt and refresh data if need to
                log_print(now(),': start ',i,'ckpt save for',name)
                nmi_2_save = list(nmi_score_history_prefix)+nmi_score_history
                np.save(project_dir+'train_nmis{}.npy'.format(name),np.array(nmi_2_save))
                l2_2_save = list(loss_history_prefix)+loss_history
                np.save(project_dir+'train_losses{}.npy'.format(name),np.array(l2_2_save))
                saver.save(sess,project_dir+"{}/step_{}.ckpt".format(name,i)) 
                log_print(now(),': finish ',i,'ckpt save for',name)
                refresh_cond = dataset_flag==2 and i%(3*i_log)==0
                if refresh_cond:
                    refresh_train_data_and_ls(dataset_flag,current_batch=current_batch) # load new dataset to ram
                    current_batch+=1
                    current_batch%=n_ebay_batches

            xs, ys = get_train_batch(dataset_flag,k,n,recompute_ys=recompute_ys,name=name) # get new batch
            feed_dict = {model.x: xs, model.y: ys}
            try:
                log_print(now(),': train iter',i,'for',name)
                #activations_tensors = sess.run(embedder.activations_dict,feed_dict=feed_dict)
                #log_print(now(),': before embed')
                #embed = sess.run(model.x_embed,feed_dict=feed_dict) # embeddding for debug. see if oom appears here.
                #log_print(now(),': after embed')
                _,clustering_history,clustering_diffs,loss,grads = sess.run([step,clusterer.history_list, clusterer.diff_history,model.loss, model.grads], feed_dict=feed_dict)
            #_,activations,parameters,clustering_history,clustering_diffs = sess.run([step,embedder.activations_dict,embedder.param_dict,model.clusterer.history_list,clusterer.diff_history], feed_dict=feed_dict) 
                clustering = clustering_history[-1]
                # ys_pred = np.matmul(clustering,clustering.T)
                # ys_pred = [[int(elem) for elem in row] for row in ys_pred] 
                nmi_score = skl_nmi(np.argmax(clustering, 1), np.argmax(ys, 1))
                log_print(now(),': after: ',nmi_score)
                nmi_score_history.append(nmi_score)
                loss_history.append(loss)
                log_print("clustring diffs:",clustering_diffs)
            except:
                log_print(now(),': error occured in train loop for: ',name)
                exc =  sys.exc_info()
                traceback.print_exception(*exc)
                print 'exception for:',name 
                pdb.set_trace()

        log_print(now(),': train_nmis:',nmi_score_history)
        return nmi_score_history,test_scores

    log_print(now(),': begin training')
    # end-to-end training:
    i_log = 100 # save ckpt every i_log iters 
    n_train_iters = 3000
    if mini:
        n_train_iters = 2000
    hyparams = [n_train_iters*i_log,k,n_,i_log]
    test_scores_e2e = []
    test_scores_ll = []
    if arg_dict['params']!="last":
        try:
            train_nmis,test_scores_e2e = train(model,hyparams,name)
        except:
            print get_tb()
            exit()
            pdb.set_trace()
    else:
        log_print(now(),': not training e2e')
        log_print(now(),': starting last-layer training')
        if arg_dict['permcovar']:
            filter_cond = lambda v: 'PermCovar' in str(v)
        else:
            filter_cond = lambda x: ("logits" in str(x)) and not ("aux" in str(x))
        permcovar_params = filter(filter_cond,tf.global_variables())
        last_layer_params = filter(filter_cond,tf.global_variables())
        if not arg_dict['permcovar']: last_layer_params.append(embedder.new_layer_w)
        model.train_step = model.optimizer.minimize(model.loss, var_list=last_layer_params) # freeze all other weights
        train_nmis,test_scores_ll = train(model,hyparams,name)
    #save_path = embedder.save_weights(sess)
    log_print(now(),': end training') 
    return train_nmis

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
if __name__ == "__main__":
    logfile = open(project_dir+'log_train.txt','a+')
    log_print(now(),': entered train:',now())
    argv = sys.argv
    arg_dict = my_parser(argv)
    gpu = arg_dict['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    #config = tf.ConfigProto(allow_soft_placement=True)
    print('Starting TF Session for',arg_dict['name'])
    #sess = tf.InteractiveSession(config=config)
    sess = tf.InteractiveSession()
    run(arg_dict)
    log_print(now(),': at end of train')

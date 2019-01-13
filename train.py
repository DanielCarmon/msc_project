import os
import os.path
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
import glob
import tensorflow as tf
import traceback
import sys
from data_api import *
from model import *
from control.dcdb import *
from tqdm import tqdm
from datetime import datetime
import pdb
import sys
import time
import inspect
import pickle
from tensorflow.python import debug as tf_debug
from sklearn.metrics import normalized_mutual_info_score as skl_nmi
import numpy as np
from utils import *

project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/'
logfile_path = project_dir+'/log_train.txt'

# configure debug print function:
if '--remote_run' in sys.argv:
    i_rr = sys.argv.index('--remote_run')
    remote_run = eval(sys.argv[i_rr+1])
else:
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
log_print(now(),': bzzzz')

data_params = None
k = None
embedder = None
clusterer = None


def run(arg_dict):
    global embedder, clusterer, data_params, k, sess # initialized as None. kept as global for access in debugger mode
    name = arg_dict['name']

    ###############
    # data config #
    ###############
    d = 299 # inception input size = [d,d]
    dataset_flag = arg_dict['dataset']
    mini = arg_dict['mini']
    log_print(now(),': started loading data for',name,'...')
    init_train_data(dataset_flag,mini=mini,name=name)
    log_print(now(),': finished loading data for',name,'...')
    list_final_clusters = [100,98,512,102] # categories per dataset
    n_final_clusters = list_final_clusters[dataset_flag] # num of clusters in dataset
    try:
        n_final_clusters = arg_dict['embed_dim']
    except:
        log_print('no embed dim specified. using default:',n_final_clusters)
    # batch configs:
    k = arg_dict['n_classes'] # clusters per batch
    n_gpu_can_handle = 100 # batch size
    n_ = n_gpu_can_handle/k # points per cluster
    n = n_*k
    data_params = [n, d]
    # ebay data configs:
    recompute_ys = dataset_flag==2 # only recompute for products dataset
    n_ebay_batches = 50 # number of preprocessed samples for products dataset
    log_print(now(),': finished data config')

    ###################
    # embedder config #
    ###################
    inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/new_inception/models/tmp/my_checkpoints/inception_v3.ckpt" # params pre-trained on ImageNet
    if 'reretrain' in arg_dict.keys() and arg_dict['reretrain']:
        inception_weight_path = project_dir+'e2e_baseline/step_30000.ckpt'
    weight_decay = arg_dict['weight_decay']
    embedder = InceptionEmbedder(inception_weight_path,num_classes=n_final_clusters,weight_decay=weight_decay) # embedding function
    if arg_dict['permcovar']: # if using permcovar layers. still experimental
        embedder_pointwise = embedder
        embedder = PermCovarEmbedder1(n_final_clusters).compose(embedder_pointwise)
    log_print(now(),': finished embedder config')

    ####################
    # clusterer config #
    ####################
    clst_module = arg_dict['cluster'] # clusterer to use
    init = arg_dict['init'] # init method for clusterer
    em_bw, gumbel_temp, softmin_bw = arg_dict['em_bw'],arg_dict['gumbel_temp'],arg_dict['softmin_bw']
    bw_params = [em_bw,gumbel_temp,softmin_bw]
    infer_covar = arg_dict['infer_covar']
    clusterer = clst_module([n,n_final_clusters], k, bw_params, n_iters=arg_dict['n_iters'],init=init,infer_covar=infer_covar)
    #pdb.set_trace()
    log_print(now(),': finished clusterer config')

    ###################
    # training config #
    ###################
    log_print(now(),': building model for',name)
    log_grads = False
    for_training = arg_dict['for_training']
    obj = arg_dict['obj']
    prepro = arg_dict['preprocess']
    lr = arg_dict['lr'] # base learning-rate
    use_tg = arg_dict['use_tg'] # use aux gradients
    model = Model(data_params, embedder, clusterer, prepro, lr, is_img=True,sess=sess,for_training=for_training,regularize=False, use_tg=use_tg,obj=obj,log_grads=log_grads) # compose clusterer on embedder and add loss function
    n_train_iters = 1000
    if mini:
        n_train_iters = 500
    i_log = 100 # save ckpt every i_log iters
    hyparams = [n_train_iters*i_log,k,n_,i_log]
    log_print(now(),': finished training config')

    ######################
    # tensorboard config #
    ######################
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Add summaries for end_points.
    end_points = embedder.activations_dict
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      #summaries.add(tf.summary.scalar('sparsity/' + end_point,
      #                                tf.nn.zero_fraction(x)))
    # Add summaries for losses.
    for loss in [model.loss]:
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
    # Add summaries for variables.
    for variable in slim.get_model_variables():
        if 'Aux' not in variable.name:
          summaries.add(tf.summary.scalar('max/'+variable.op.name, tf.reduce_max(variable)))
          summaries.add(tf.summary.scalar('min/'+variable.op.name, tf.reduce_min(variable)))
          summaries.add(tf.summary.scalar('mean/'+variable.op.name, tf.reduce_mean(variable)))
          #with tf.name_scope('std_dev') as scope:
          #    std_dev = tf.sqrt(tf.reduce_mean(tf.square(variable - tf.reduce_mean(variable))))
          #summaries.add(tf.summary.scalar('std_dev/'+variable.op.name,std_dev))
          summaries.add(tf.summary.histogram('variables/' + variable.op.name, variable))
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    tensorboard_log_dir = project_dir+'tb_log_dir'+name
    train_writer = tf.summary.FileWriter(tensorboard_log_dir + '/train', sess.graph)
    log_print(now(),': finished tensorboard config')

    ######################
    # checkpoints config #
    ######################
    sess.run(tf.global_variables_initializer())
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
            with tf.device('/cpu:0'):
                saver.restore(sess,ckpt_path)
            nmi_score_history_prefix = np.load(project_dir+'train_nmis{}.npy'.format(name))
            loss_history_prefix = np.load(project_dir+'train_losses{}.npy'.format(name))
    except:
        log_print(now(),': no previous checkpoints found for',name,'; loading from ImageNet ckpt')
        embedder.load_weights(sess)
        nmi_score_history_prefix = []
        loss_history_prefix = []
    log_print(now(),': finished checkpoints config')

    def train(model,hyparams,name): # training procedure
        global test_scores_em,test_scores_km # global so it could be reached at debug pm mode
        test_scores = []
        n_steps,k,n_,i_log  = hyparams
        i_tensorboard = 100
        loss_history = []
        nmi_score_history = []
        step = model.train_step # update step op
        current_batch = 0 # only relevant for ebay dataset
        for i in range(n_offset,n_steps): # main loop
            xs, ys = get_train_batch(dataset_flag,k,n,recompute_ys=recompute_ys,name=name) # new batch
            feed_dict = {model.x: xs, model.y: ys}
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
            try:
                tic = now()
                log_print(tic,': train iter',i,'for',name)
                if i%i_tensorboard==0: # save summaries for tensorboard
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _,summary,clustering_history,clustering_diffs,loss,grads = sess.run([step,summary_op,clusterer.history_list, clusterer.diff_history,model.loss, model.grads], options=run_options, run_metadata=run_metadata,feed_dict=feed_dict)

                    summary = sess.run(summary_op,feed_dict)
                    train_writer.add_summary(summary,i)
                    train_writer.add_run_metadata(run_metadata,'step'+str(i))
                else:
                    _,clustering_history,clustering_diffs,loss,grads = sess.run([step,clusterer.history_list, clusterer.diff_history,model.loss, model.grads], feed_dict=feed_dict)

                clustering = clustering_history[-1]
                nmi_score = skl_nmi(np.argmax(clustering, 1), np.argmax(ys, 1))
                toc = now()
                log_print(toc,': after: ',nmi_score)
                log_print('elapsed:',toc-tic)
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
        train_writer.close()
        return nmi_score_history,test_scores

    ######################
    # kickstart training #
    ######################
    log_print(now(),': begin training')
    test_scores_e2e = []
    test_scores_ll = []
    if arg_dict['params']!="last":
        try:
            train_nmis,test_scores_e2e = train(model,hyparams,name)
        except:
            print get_tb()
            exit()
    else:
        log_print(now(),': last-layer training')
        if arg_dict['permcovar']:
            filter_cond = lambda v: 'PermCovar' in str(v)
        else:
            filter_cond = lambda x: ("Logits" in str(x))
        last_layer_params = filter(filter_cond,tf.global_variables())
        model.train_step = model.optimizer.minimize(model.loss, var_list=last_layer_params) # freeze all other weights
        train_nmis,test_scores_ll = train(model,hyparams,name)
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
    exit()

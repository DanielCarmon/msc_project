import os
import os.path
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
os.environ["OMP_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
import tensorflow as tf
from sklearn import cluster
import traceback
import sys
from data_api import *
from model import *
from dcdb import *
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
    #pdb.set_trace()
    exc = sys.exc_info()
    return traceback.print_exception(*exc)

def trim(vec, digits=3):
    factor = 10 ** digits
    vec = np.round(vec * factor) / factor
    return vec


# Experiments to test refractored model.py code
# START
# from pylab import *
import matplotlib.pyplot as plt
project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/'
rand = np.random.normal

data_params = None
k = None
embedder = None
clusterer = None
tf_clustering = None


def init1():
    global embedder, clusterer, tf_clustering, data_params, k
    # k = 3 # num of rectangles to sample
    k = 3
    # d = 28 # img dim
    d = 40
    data_params = [d, 3]
    embedder = ImgEmbedder(data_params)
    clusterer = GDKMeansClusterer2([d ** 2, 3], k + 1)
    # clusterer = EMClusterer([d**2,3],k+1,n_iters=50)
    tf_clustering = clusterer.infer_clustering()


def run1():
    global embedder, clusterer, tf_clustering, data_params, k
    d = data_params[0]
    xs, ys = get_img(1, k, d)
    x, y = xs[0], ys[0]  # (d,d,3) array , list of len==k
    bg = get_background_mask(y)
    bg_ = np.reshape(bg, (d ** 2, 1))
    # pdb.set_trace()
    y_old = combine_masks(y)
    y = y_old + bg_ * bg_.T
    # x = 10*x
    x = noisify(x)
    # x = 10*x
    x_new = sess.run(embedder.x_new, feed_dict={embedder.x: x})
    # pdb.set_trace()
    print('meow')
    x_new = noisify(x_new)
    scatter_3d(x_new)
    # plt.show()
    pdb.set_trace()
    # x_new = 30*x_new
    # x_new = noisify(x_new)
    # x_new = 10*x_new
    # x_new = noisify(x_new)
    # clusterer = GDKMeansClusterer2(data_params,k)
    # tf_clustering = clusterer.infer_clustering()
    tf_membership = clusterer.history_list
    feed_dict = {clusterer.x: x_new}
    [last_centroids, clustering_history, cost_history, membership_history, grad_log] = sess.run(
        [clusterer.c, tf_clustering, clusterer.cost_log, clusterer.history_list, clusterer.grad_log],
        feed_dict=feed_dict)
    # [clustering_history,membership_history] = sess.run([tf_clustering,tf_membership],feed_dict=feed_dict) # for em
    # cost_history = []
    last_cluster = clustering_history[-1]
    last_membership = membership_history[-1]
    more = [last_membership, last_centroids, cost_history]
    plotup(clustering_history, x_new, y, more)


def plotup(clustering, x, y, more):
    import matplotlib.animation as anim
    last_membership, last_centroids, _ = more
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    img_last = Image.fromarray(clustering[-1] * 255)
    plt.title('last prediction')
    plt.imshow(img_last)
    ax1 = fig.add_subplot(2, 2, 3)
    ims = []
    for time in xrange(np.shape(clustering)[0]):
        im = ax1.imshow(clustering[time, :, :], cmap='gray')
        ims.append([im])
    # run animation
    ani = anim.ArtistAnimation(fig, ims, interval=100, blit=False)
    plt.title('predicted clustering trajectory')
    fig.add_subplot(2, 2, 2)
    img = Image.fromarray(y * 255)
    plt.imshow(img)
    plt.title('ground truth clustering')
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    # show 3d scatter of data vectors:
    # pdb.set_trace()
    indices = np.argmax(last_membership, 1)
    # indices = np.array(indices)
    cs = ['r', 'b', 'g', 'k', 'c']
    for i in range(4):
        c = cs[i]
        points = x[indices == i]
        xs, ys, zs = list(points[:, 0]), list(points[:, 1]), list(points[:, 2])
        ax.scatter(xs, ys, zs, c=c, marker='o')
    # plot centroid:
    c = cs[4]
    xs, ys, zs = list(last_centroids[:, 0]), list(last_centroids[:, 1]), list(last_centroids[:, 2])
    ax.scatter(xs, ys, zs, c=c, marker='o')
    # plt.imshow(x)
    plt.title('data')
    # plt.xlim((-20,20))
    # plt.ylim((-20,20))
    # plt.zlim((-20,20))
    # plt.plot(cost_log)
    # plt.title('cost trajectory')
    last_clustering = clustering[-1]
    # print 'Avg. norm of difference:',np.linalg.norm(last-y)/(y.shape[0])**2
    print 'Norm of difference:', np.linalg.norm(last_clustering - y)
    print "Sanity check:", np.linalg.norm(last_clustering.T - last_clustering), np.linalg.norm(y.T - y)
    ts = str(datetime.today())
    # ani.save('./animation{}.gif'.format(ts),writer='imagemagick',fps=10)
    # fig.save('/tmp/fig.gif', writer='imagemagick', fps=30)
    print 'meow1'
    """
    pdb.set_trace()
    for i in range(784):
        if np.linalg.norm(y[i]-last_clustering[i])>1e-10:
            print 'at disagreement'
            print i,list(x[i]),last_centroids[indices[i]],indices[i]
            #pdb.set_trace()
    x_round = np.round(x)
    for i in range(784):
        for j in range(784):
            norm = np.linalg.norm(x_round[i]-x_round[j])
            sim = norm<0.5
            same_clst = bool(y[i][j])
            if sim!=same_clst:
                print False,i,j
    #    #print list(y[i])
    #    #print list(last_clustering[i])
    """
    try:
        plt.show()
    except AttributeError:
        pdb.set_trace()
    print 'meow2'


def run2():
    global embedder, clusterer, tf_clustering, data_params, k
    n, d = 4 * 10, 3
    k = 4
    data_params = 4 * n, d
    np.random.seed(2018)
    v1 = np.array([0, 0, 0])[np.newaxis, :]
    # v1 = np.random.rand(1,3)
    x1 = np.tile(v1, [2 * n + n / 4, 1])
    v2 = np.array([1, 0, 0])[np.newaxis, :]
    # v2 = np.random.rand(1,3)
    x2 = np.tile(v2, [n / 4, 1])
    v3 = np.array([0, 1, 0])[np.newaxis, :]
    # v3 = np.random.rand(1,3)
    x3 = np.tile(v3, [n / 4, 1])
    v4 = np.array([0, 0, 1])[np.newaxis, :]
    # v4 = np.random.rand(1,3)
    x4 = np.tile(v4, [n + n / 4, 1])

    y1 = np.hstack((np.ones((1, 2 * n + n / 4)), np.zeros((1, n + 3 * n // 4))))
    y2 = np.hstack((np.zeros((1, 2 * n + n / 4)), np.ones((1, n / 4))))
    y2 = np.hstack((y2, np.zeros((1, n + n / 2))))
    y3 = np.hstack((np.zeros((1, 2 * n + n / 2)), np.ones((1, n / 4))))
    y3 = np.hstack((y3, np.zeros((1, n + n / 4))))
    y4 = np.hstack((np.zeros((1, 2 * n + 3 * n / 4)), np.ones((1, n + n / 4))))
    y = y1 * y1.T + y2 * y2.T + y3 * y3.T + y4 * y4.T
    # pdb.set_trace()

    x1 = np.vstack((x1, x2))
    x2 = np.vstack((x3, x4))
    x = np.vstack((x1, x2))
    # perm = np.random.permutation(np.eye(4*n))
    # x = np.matmul(perm,x)
    # y = np.matmul(perm,np.matmul(y,perm.T))
    ''
    # clusterer = EMClusterer(data_params,k)
    clusterer = GDKMeansClusterer2(data_params, k)
    tf_clustering = clusterer.infer_clustering()
    tf_membership = clusterer.history_list
    feed_dict = {clusterer.x: x}
    [clustering_history, cost_history, membership_history, grad_log] = sess.run(
        [tf_clustering, clusterer.cost_log, clusterer.history_list, clusterer.grad_log], feed_dict=feed_dict)
    # [clustering_history,membership_history] = sess.run([tf_clustering,tf_membership],feed_dict=feed_dict) # for em
    # cost_history = []
    plotup(clustering_history, x, y, cost_history)
    last_cluster = clustering_history[-1]
    last_membership = membership_history[-1]

    pdb.set_trace()


def run3():
    # check gradient flow through clusterers.
    # dataset =  ambiguous gaussians
    global embedder, clusterer, tf_clustering, data_params, k

    k = 2
    n, d = 200,2
    data_params = 4*n, d
    def gen_xy():
        shift,r = np.random.normal([0,0]),np.random.normal([5])
        flip = (-1)**np.random.randint(2)
        x1 = np.random.normal([0,0],size=[n,2])
        x2 = np.random.normal([0,1],size=[n,2])
        x3 = np.random.normal([1,0],size=[n,2])
        x4 = np.random.normal([1,1],size=[n,2])
        x =  np.vstack((np.vstack((x1,x2)),np.vstack((x3,x4))))
        x = flip*r*(x+shift)
        y1 = np.array([1]*2*n+[0]*2*n)
        y2 = np.array([0]*2*n+[1]*2*n)
        y = np.vstack((y1,y2)) # [4n,2]
        y = np.matmul(y.T,y) # [4n,4n]
        return x,y
    embedder = ProjectionEmbedder(data_params)

    embed_dim = 1
    clusterer = EMClusterer([4*n,embed_dim],k,n_iters=10)
    model = Model(data_params, embedder, clusterer)

    param_history = []
    loss_history = []
    nmi_history = []
    step = model.train_step
    sess.run(tf.global_variables_initializer())
    n_steps = 15000
    for i in range(n_steps):
        print 'at train step', i
        x,y = gen_xy()
        feed_dict = {model.x: x, model.y: y}
        _, membership_log, loss, diff_history = sess.run([step, clusterer.history_list, model.loss,clusterer.diff_history], feed_dict=feed_dict)
        #print 'diff history:',diff_history
        loss_history.append(loss)
        param_history.append(sess.run(embedder.params))
        last_membership = membership_log[-1]
        indices = np.argmax(last_membership, 1)
        nmi_score = nmi(indices, np.argmax(y, 1))
        nmi_history.append(nmi_score)
    history = [loss_history,nmi_history]
    save(history,'gaussians_score')
    """ # plot 
    plt.plot(loss_history)
    plt.title('loss')
    title = "Clustered Data"
    scatter_3d(x, indices, title)
    title = "Projection Vector Updates"
    scatter_3d(np.reshape(param_history, (n_steps, 3)), title=title)
    # scatter_3d()
    print 'last projection:', param_history[-1]
    """
def run4(arg_dict):
    global embedder, clusterer, tf_clustering, data_params, k, sess
    d = 299
    k = 2
    if 'n_train_classes' in arg_dict.keys():
        k = arg_dict['n_train_classes']
    dataset_flag = arg_dict['dataset']
    n_gpu_can_handle = [100,100,100][dataset_flag]
    n_ = n_gpu_can_handle/k # points per cluster
    n = n_*k
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
    list_final_clusters = [100,98,512]
    n_final_clusters = list_final_clusters[dataset_flag] # num of clusters in dataset
    embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim,new_layer_width=n_final_clusters)
    if arg_dict['deepset']:
        embedder_pointwise = embedder
        embedder = DeepSetEmbedder1(embed_dim).compose(embedder_pointwise) # Under Construction!
    clusterer = clst_module([n, embed_dim], k, hp, n_iters=arg_dict['n_iters'],init=init)
    print 'building model object'
    obj = arg_dict['obj']
    model = Model(data_params, embedder, clusterer, model_lr, is_img=True,sess=sess,for_training=False,regularize=False, use_tg=use_tg,obj=obj)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
    n_offset = 0 # no. of previous checkpoints
    try: # restore last ckpt
        if arg_dict['restore_last']:
            name = arg_dict['name']
            ckpt_path = project_dir+name
            n_ckpt_files  = len([fname for fname in os.listdir(ckpt_path) if os.path.isfile(os.path.join(ckpt_path, fname))])
            n_last_ckpt = ((n_ckpt_files-1)/3-1)*100
            ckpt_path = ckpt_path+'/step_{}.ckpt'.format(n_last_ckpt)
            print 'Restoring parameters from',ckpt_path
            saver.restore(sess,ckpt_path)
            n_offset = n_last_ckpt+1
            name = arg_dict['name']
            nmi_score_history_prefix = np.load(project_dir+'train_nmis{}.npy'.format(name))
            loss_history_prefix = np.load(project_dir+'train_losses{}.npy'.format(name))
    except: 
        print 'no previous checkpoints found'
        nmi_score_history_prefix = []
        loss_history_prefix = []

    def train(model,hyparams):
        global test_scores_em,test_scores_km # global so it could be reached at debug pm mode
        test_scores = []
        train_scores = []
        n_steps,k,n_,i_log  = hyparams
        param_history = []
        loss_history = []
        nmi_score_history = []
        step = model.train_step

        debug = False
        for i in range(n_offset,n_steps): 
            xs, ys = get_train_batch(dataset_flag,k,n)
            feed_dict = {model.x: xs, model.y: ys}
            print 'at train step', i
            if (i%i_log==0): # case where i==0 is baseline
                name = arg_dict['name']
                print 'meow'
                nmi_2_save = list(nmi_score_history_prefix)+nmi_score_history
                np.save(project_dir+'train_nmis{}.npy'.format(name),np.array(nmi_2_save))
                l2_2_save = list(loss_history_prefix)+loss_history
                np.save(project_dir+'train_losses{}.npy'.format(name),np.array(l2_2_save))
                saver.save(sess,project_dir+"{}/step_{}.ckpt".format(name,i)) 
                print 'woem'
            try:
                print 'updating parameters'
                for i in range(1):
                    #pdb.set_trace()
                    #activations_tensors = sess.run(embedder.activations_dict,feed_dict=feed_dict)
                    #print 'before embed'
                    #embed = sess.run(model.x_embed,feed_dict=feed_dict) # embeddding for debug. see if oom appears here.
                    #print 'after embed'
                    _,clustering_history,clustering_diffs,loss,grads = sess.run([step,clusterer.history_list, clusterer.diff_history,model.loss, model.grads], feed_dict=feed_dict)
                #_,activations,parameters,clustering_history,clustering_diffs = sess.run([step,embedder.activations_dict,embedder.param_dict,model.clusterer.history_list,clusterer.diff_history], feed_dict=feed_dict) 
            except:
                print 'error occured'
                exc =  sys.exc_info()
                traceback.print_exception(*exc)
                time.sleep(5)
                print 'exiting main.py'
                #exit()
                #pdb.set_trace()
            clustering = clustering_history[-1]
            # ys_pred = np.matmul(clustering,clustering.T)
            # ys_pred = [[int(elem) for elem in row] for row in ys_pred] 
            nmi_score = nmi(np.argmax(clustering, 1), np.argmax(ys, 1))
            print 'after: ',nmi_score
            nmi_score_history.append(nmi_score)
            loss_history.append(loss)
            print "clustring diffs:",clustering_diffs
            if debug: 
                pdb.set_trace()
        print 'train_nmis:',nmi_score_history
        return nmi_score_history,test_scores

    print 'begin training'
    # end-to-end training:
    i_log = 100 
    n_train_iters = 3500
    hyparams = [n_train_iters*i_log,k,n_,i_log]
    test_scores_e2e = []
    test_scores_ll = []
    if arg_dict['deepset']:
        filter_cond = lambda x: 'DeepSet' in str(x)
        deepset_params = filter(filter_cond,tf.global_variables())
        model.train_step = model.optimizer.minimize(model.loss, var_list=deepset_params) # freeze inception weights
        train_nmis,test_scores = train(model,hyparams)
        print 'finished training'
    if arg_dict['train_params']!="last":
        try:
            train_nmis,test_scores_e2e = train(model,hyparams)
        except:
            print get_tb()
            exit()
            pdb.set_trace()
    else:
        print 'not training e2e'
        print 'starting last-layer training'
        # last-layer training (use this in case of overfitting):
        hyparams[0]=500*i_log
        filter_cond = lambda x: ("logits" in str(x)) and not ("aux" in str(x))
        #filter_cond = lambda x: ("aux_logits/FC/" in str(x))
        last_layer_params = filter(filter_cond,embedder.params)
        last_layer_params.append(embedder.new_layer_w)
        model.train_step = model.optimizer.minimize(model.loss, var_list=last_layer_params) # freeze all other weights
        train_nmis,test_scores_ll = train(model,hyparams)
    #save_path = embedder.save_weights(sess)
    print 'end training' 
    return train_nmis
def run5(dataset_flag=0,output_layer='logits'):
    """ test Inception baseline for clustering bird classes 101:200 """
    global sess
    d = 299
    split_flag = 1
    pdb.set_trace()
    data = get_data(split_flag,dataset_flag) 
    n = data[0].shape[0]
    xs,ys,ys_membership = data
    n_clusters = ys_membership.shape[1]
    
    data_params = [n, d]
    inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
    embed_dim = 1001
    embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim,output_layer=output_layer)

    tf_x = tf.placeholder(tf.float32, [None, d, d, 3])
    print 'building embedding pipeline'
    tf_endpoint = embedder.embed(tf_x)
    print 'running global_variables_initializer()'
    sess.run(tf.global_variables_initializer())
    embedder.load_weights(sess)
    def get_embedding(xs_batch,tf_endpoint):
        feed_dict = {tf_x:xs_batch}
        return sess.run(tf_endpoint,feed_dict=feed_dict)
    np_embeddings = np.zeros((0,embed_dim))
    i=0
    while 60*i<n:
        xs_batch = xs[60*i:60*(i+1)]
        print 'embedding batch ',i
        embedded_xs_batch = get_embedding(xs_batch,tf_endpoint)
        np_embeddings = np.vstack((np_embeddings,embedded_xs_batch))
        i+=1
    print xs.shape
    print np_embeddings.shape
    #np_embeddings1001 = np_embeddings[:,:]
    #neuron_inds = range(2048)
    #subsampled_inds = np.random.choice(neuron_inds,n_clusters)
    #np_embeddings = np_embeddings[:,subsampled_inds]  # subsampling
    pdb.set_trace()
    np_embeddings_normalized = l2_normalize(np_embeddings)
    n = np_embeddings.shape[0]
    from sklearn import cluster
    KMeans = cluster.KMeans
    print 'begin kmeans fit over embeddings'
    #km = KMeans(n_clusters=n_clusters,init='random').fit(np_embeddings) # regular kmeans
    km = KMeans(n_clusters=n_clusters).fit(np_embeddings)
    print 'endkmeans fit over embeddings'
    print 'begin kmeans fit over normalized embeddings'
    km_normalized = KMeans(n_clusters=n_clusters).fit(np_embeddings_normalized)
    print 'end kmeans fit over normalized embeddings'
    import pickle
    f = open('sklearn_kmeans_res.txt','w')
    labels = km.labels_
    labels_normalized = km_normalized.labels_
    nmi_score = nmi(labels, np.argmax(ys_membership, 1))
    nmi_score_normalized = nmi(labels_normalized, np.argmax(ys_membership, 1))
    scores = [nmi_score,nmi_score_normalized]
    pickle.dump(scores,f)
    print scores
    pdb.set_trace()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
if __name__ == "__main__":
    print 'start logginggggg'
    argv = sys.argv
    run = argv[1]
    if run=='3':
        run3()
    if run=='4':
        if len(argv)>2:
            arg_dict = my_parser(argv)
        gpu = arg_dict['gpu']
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        #config = tf.ConfigProto(allow_soft_placement=True)
        print('Starting TF Session')
        #sess = tf.InteractiveSession(config=config)
        sess = tf.InteractiveSession()
        run4(arg_dict)
    if run=='5':
        dataset_flag=0
        if len(argv)>2:
            dataset_flag=int(argv[2])
            if len(argv)>3:
                output_layer=argv[3]
        run5(dataset_flag,output_layer)

'''
train_nmis,test_nmis = [],[]
try:
    train_nmis,test_nmis = run4()
except:
    exc =  sys.exc_info()
    traceback.print_exception(*exc)
    pdb.set_trace()
tf.reset_default_graph()
print train_nmis,test_nmis
'''
print 'finish'
"""
# plot approximately how good is each hypothesis.
#from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
rand = np.random.normal
rect_len = 30

n = 50
#x,y_onehot = get_unfaithfull_data(n,r=1)
x,y_onehot = get_gaussians(n)
scatter(x,np.argmax(y_onehot,1))
#print(y_onehot)
y = get_clst_mat(y_onehot,'one-hot')
method = "em"
model = Model(num_clst=2,d=2*n,num_col=2,method=method)
print('Starting TF Session')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

hyp_range = 5
ls = np.linspace(-hyp_range,hyp_range,rect_len)

#p = ax.plot_surface(hypX,hypY,hypZ,rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#cb = fig.colorbar(p, shrink=0.5)
hypX,hypY = np.meshgrid(ls,ls)
hypZ = np.zeros((rect_len,rect_len))
for i in range(rect_len):
    for j in range(rect_len):
        w = np.array([ls[i],ls[j]])
        #w = np.array([1,0])
        w = np.reshape(w,[2,1])
        #print i,j,w
        #Model.params = tf.constant(w)
        #print 'meow1,',sess.run(model.params)
        
        tmp = sess.run(model.loss,feed_dict={model.x_:x,model.y_:y,model.params:w})
        #print 'meow2,',sess.run(model.params)
        relevant_loss = tmp[len(tmp)-1]/((2*n)**2)
        #relevant_loss = tmp[len(tmp)-1]
        print 'iter',i,j,'loss:',relevant_loss
        hypZ[i,j]=relevant_loss
print('finished looping over hypotheses')
#pdb.set_trace()
y_preds,np_x_embed = sess.run([model.y_preds,model.x_embed],feed_dict={model.x_:x,model.params:w})
y_pred = y_preds[len(y_preds)-1]
x_2d = np.hstack((np_x_embed,np.zeros((2*n,1))))
#scatter(x_2d,np.argmax(y_onehot,1))
plt.imshow(hypZ)
plt.colorbar()
plt.xticks(list(range(rect_len)),trim(hypX[0,:]))
plt.xlabel('x coordinate of projection vector')
plt.yticks(list(range(rect_len)),trim(hypY[:,0]))
plt.ylabel('y coordinate of projection vector')
plt.title('Loss when clustering under different projections')
plt.show()
'''
fig = plt.figure(figsize=(10*rect_len,10*rect_len))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_xticks(ls)
ax.set_yticks(ls)
ax.set_xlabel('x-coordinate of projection')
ax.set_ylabel('y-coordinate of projection')
#p = ax.plot_wireframe(hypX,hypY,hypZ)
p = ax.plot_surface(hypX,hypY,hypZ,rstride=1, cstride=1,cmap='hot')
plt.title('Loss at different hypotheses')
timestamp = datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M%p")
plt.savefig("loss_surface_figure_{}".format(timestamp))
plt.show()
'''
"""
"""
# see whether network is able to learn linear transformation
n = 100
x,y_onehot = get_unfaithfull_data(n,1)
#scatter(x,np.argmax(y_onehot,1))
y = get_clst_mat(y_onehot,'one-hot')
method = "em"
model = Model(num_clst=2,d=4*100,num_col=2,method=method)
print('Starting TF Session')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
n_train_steps = 200
for i in range(n_train_steps):
    # train
    print("enter train step:",i)
    sess.run(model.train_step,feed_dict={model.x_:x,model.y_:y})
    print("model params after train step #",i,':',sess.run(model.params))
np_x_embed = sess.run(model.x_embed,feed_dict={model.x_:x})
x_2d = np.hstack((np_x_embed,np.zeros((4*n,1))))
scatter(x_2d,np.argmax(y_onehot,1))
"""
"""
# show that network is able to fit mixture of k-gaussians with given k
n = 100 
k = 5
x = get_gmm_data(n,k)
indices = np.arange(k)
indices = np.repeat(indices,[n]*k)
scatter(x,indices)

method='em'
model = Model(num_clst=k,d=n*k,method=method)
print('Starting TF Session')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
x_embed,np_bs = sess.run([model.x_embed,model.bs],feed_dict={model.x_:x})
b_last = np_bs[model.NUM_OPT_STEPS-1]
predicted_indices = np.argmax(b_last,axis=1)
scatter(x,predicted_indices)
"""
"""
k = 1 # num of rectangles to sample
d = 28 # img dim
xs,ys = get_img(1,k,d)
x,y = xs[0],ys[0] # (d,d,3) array , list of len==k
x = noisify(x)
method='em'
pdb.set_trace()

model = Model(num_clst=k+1,d=d,method=method,img_input=True)
print('Starting TF Session')
sess = tf.InteractiveSession()

x_embed,np_bs = sess.run([model.x_embed,model.bs],feed_dict={model.x_:x})
pdb.set_trace()
b_last = np_bs[model.NUM_OPT_STEPS-1]
indices = np.argmax(b_last,axis=1)
# calculate disagreement with gt:
clst_mat_pred = np.matmul(b_last,(b_last.T))

def get_obj_val(x_embed,y):
    img_len = y[0].shape[0]
    b = np.ndarray((img_len**2,0))
    for mask in y:
        # mask is [d,d,3]
        mask_ = np.sum(mask,axis=2)
        vec = np.reshape(mask_,(img_len**2,1))
        b = np.hstack((b,vec))
    clst_sz = np.sum(b,axis=0) # == tf.transpose(b)*tf.ones([n,1])
    print('cluster sizes:')
    print clst_sz
    inv_sz = np.linalg.inv(np.diag(clst_sz))
    matmul_tmp = np.matmul(inv_sz,(b.T))
    centroid_matrix = np.matmul(matmul_tmp,x_embed)
    return np.linalg.norm(np.matmul(b,centroid_matrix)-x_embed)

print 'Displaying Image'
display(x)
print 'Analyzing results...'
clst_mat_gt = get_clst_mat(y)
#pdb.set_trace()
# todo: replace by np.einsum, or get values from tf graph
diffs = np.array([np.linalg.norm(clst_mat_gt-np.matmul(b,(b.T))) for b in tqdm(np_bs)])
print 'Done'
# display results:
gs = gridspec.GridSpec(2, 4)
ax1 = plt.subplot(gs[0,0:2])
print('meow2')
img = Image.fromarray(clst_mat_pred*255)
ax1.imshow(img)
ax1.set_title('Predicted Clustering')
ax2 = plt.subplot(gs[0,2:])
img = Image.fromarray(clst_mat_gt*255)
ax2.imshow(img)
ax2.set_title('Ground Truth Clustering')
ax3 = plt.subplot(gs[1,1:3])
ax3.set_title('K-Means Clustering via Gradient Descent:')
plt.plot(diffs)
plt.xlabel('Iteration')
plt.ylabel('Distance from ground truth')
plt.show()


x_embed = x_embed[:,2:]
x_embed = noisify(x_embed)
#scatter(d,indices)
"""

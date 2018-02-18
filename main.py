import tensorflow as tf
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
import inspect
from tensorflow.python import debug as tf_debug
from sklearn.metrics import normalized_mutual_info_score as nmi

def linenum():
    """ Returns current line number """
    return inspect.currentframe().f_back.f_lineno


def trim(vec, digits=3):
    factor = 10 ** digits
    vec = np.round(vec * factor) / factor
    return vec


# Experiments to test refactored model.py code
# START
# from pylab import *
import matplotlib.pyplot as plt

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
    global embedder, clusterer, tf_clustering, data_params, k

    k = 3
    n, d = k * 100, 3
    data_params = n, d
    r = 4  # distance between gaussians
    rashit = np.ones(d)
    x1 = np.random.normal(0 * rashit, 1, (n / k, d))
    x2 = np.random.normal(r * rashit, 1, (n / k, d))
    x3 = np.random.normal((2 * r) * rashit, 1, (n / k, d))
    x = np.vstack((x1, x2))
    x = np.vstack((x, x3))
    y1 = np.array([1.] * (n / 3) + [0.] * (2 * n / 3))[np.newaxis, :]
    y2 = np.array([0.] * (n / 3) + [1.] * (n / 3) + [0.] * (n / 3))[np.newaxis, :]
    y3 = np.array([0.] * (2 * n / 3) + [1.] * (n / 3))[np.newaxis, :]
    y = np.zeros((n, n))
    y += np.matmul(y1.T, y1)
    y += np.matmul(y2.T, y2)
    y += np.matmul(y3.T, y3)

    embedder = ProjectionEmbedder(data_params)

    d_embed = 1
    clusterer = GDKMeansClusterer2((n, d_embed), k)
    # clusterer = EMClusterer((n,d_embed),k)
    model = Model(data_params, embedder, clusterer)

    param_history = []
    loss_history = []
    step = model.train_step
    feed_dict = {model.x: x, model.y: y}
    sess.run(tf.global_variables_initializer())
    n_steps = 150
    for i in range(n_steps):
        print 'at train step', i
        _, membership_log, loss = sess.run([step, clusterer.history_list, model.loss], feed_dict=feed_dict)
        loss_history.append(loss)
        param_history.append(sess.run(embedder.params))
    last_membership = membership_log[-1]
    indices = np.argmax(last_membership, 1)
    plt.plot(loss_history)
    plt.title('loss')
    title = "Clustered Data"
    scatter_3d(x, indices, title)
    title = "Projection Vector Updates"
    scatter_3d(np.reshape(param_history, (n_steps, 3)), title=title)
    # scatter_3d()
    print 'last projection:', param_history[-1]

def run4():
    global embedder, clusterer, tf_clustering, data_params, k, sess
    d = 299
    k = 2
    n_ = 30 # points per cluster
    data_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds'
    n = n_*k
    data_params = [n, d]
    inception_weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3"
    vgg_weight_path = '/specific/netapp5_2/gamir/carmonda/research/vision/vgg16_weights.npz'
    #weight_path = '/home/d/Desktop/uni/research/vgg16_weights.npz'
    #embed_dim = 128
    # embedder = Vgg16Embedder(vgg_weight_path,sess=sess,embed_dim=embed_dim)
    embed_dim = 1001
    embedder = InceptionEmbedder(inception_weight_path,embed_dim=embed_dim)
    clusterer = EMClusterer([n, embed_dim], k, n_iters = 5)
    model = Model(data_params, embedder, clusterer, is_img=True,sess=sess)
    
    hyparams = 50,data_dir,k,n_

    def train(model,hyparams):
        n_steps,data_dir,k,n_ = hyparams
        param_history = []
        loss_history = []
        nmi_score_history = []
        step = model.train_step

        debug = False
        for i in range(n_steps): 
            xs, ys = get_bird_train_data2(data_dir,k, n_)
            feed_dict = {model.x: xs, model.y: ys}
            print 'at train step', i
            try:
                _,tensor1,tensor2,clustering_history,clustering_diffs = sess.run([step,embedder.activations_dict,embedder.param_dict,model.clusterer.history_list,clusterer.diff_history], feed_dict=feed_dict)
            except:
                print 'error occured'
                exc =  sys.exc_info()
                traceback.print_exception(*exc)
                pdb.set_trace()
            print "clustring diffs:",clustering_diffs
            clustering = clustering_history[-1]
            # ys_pred = np.matmul(clustering,clustering.T)
            # ys_pred = [[int(elem) for elem in row] for row in ys_pred] 
            nmi_score = nmi(np.argmax(clustering, 1), np.argmax(ys, 1))
            print 'nmi_score: ',nmi_score
            nmi_score_history.append(nmi_score)
            if debug: 
                pdb.set_trace()
        print nmi_score_history
    # end-to-end training:
    train(model,hyparams)
    # last-layer training:
    #last_layer_params = filter(lambda x: ("logits" in str(x)) and not ("aux" in str(x)),embedder.params)
    #model.train_step = model.optimizer.minimize(model.loss, var_list=last_layer_params) # freeze all other weights
    #train(model,hyparams)

    # test
    # averagce over many of them?
    xs_test, ys_test = get_bird_test_data2(data_dir,k, n_)
    feed_dict = {model.x: xs_test,model.y:ys_test}
    clustering,loss = sess.run([model.clusterer.history_list,model.loss],feed_dict=feed_dict)
    clustering = clustering[-1]
    nmi_score = nmi(np.argmax(clustering, 1), np.argmax(ys_test, 1))
    print 'test loss:',loss
    print 'test nmi:',nmi_score
    return nmi_score
def run5():
    global embedder, clusterer, tf_clustering, data_params, k, sess
    d = 224
    k = 5
    xs, ys = get_bird_train_data(k, d)
    n = xs.shape[0]
    data_params = [n, d, d, 3]
    pdb.set_trace()

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
res = []
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
print('Starting TF Session')
sess = tf.InteractiveSession(config=config)
try:
    res.append(run4())
except:
    pdb.set_trace()
tf.reset_default_graph()
print res
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

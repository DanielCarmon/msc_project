import tensorflow as tf
from data_api import *
#from model import Model
from model import *
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from tqdm import tqdm
import datetime
import pdb
import inspect
from tensorflow.python import debug as tf_debug
def linenum():
    """ Returns current line number """
    return inspect.currentframe().f_back.f_lineno
def trim(vec,digits=3):
    factor = 10**digits
    vec = np.round(vec*factor)/factor
    return vec

# Experiments to test refactored model.py code
# START
#from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
rand = np.random.normal

data_params = None
k = None
embedder = None
clusterer = None
tf_clustering = None
def init1():
    global embedder,clusterer,tf_clustering,data_params,k
    k = 3 # num of rectangles to sample
    d = 28 # img dim
    data_params = [d,3]
    embedder = ImgEmbedder(data_params)
    #clusterer = GDKMeansClusterer([d**2,3],k+1,n_iters=150)
    clusterer = EMClusterer([d**2,3],k+1,n_iters=50)
    tf_clustering = clusterer.infer_clustering()
def run1():
    global embedder,clusterer,tf_clustering,data_params,k
    d = data_params[0]
    xs,ys = get_img(1,k,d)
    x,y = xs[0],ys[0] # (d,d,3) array , list of len==k
    bg = get_background_mask(y)
    bg_ = np.reshape(bg,(d**2,1))
    y = combine_masks(y)
    y+=bg_*bg_.T
    x_new = sess.run(embedder.x_new,feed_dict={embedder.x:x})
    x_new = noisify(x_new)
    #clusterer = EMClusterer(x_new.shape,k+1)
    clustering = sess.run(tf_clustering,feed_dict={clusterer.x:x_new})
    last = clustering[-1]
    #print 'Norm of difference:',np.linalg.norm(last-y)/(y.shape[0])**2
    print 'Norm of difference:',np.linalg.norm(last-y)
    plotup(clustering,x,y)
def plotup(clustering,x,y,cost_log):
    import matplotlib.animation as anim
    import types

    fig = plt.figure()
    fig.add_subplot(2,2,1)
    img_last = Image.fromarray(clustering[-1]*255)
    plt.title('last prediction')
    plt.imshow(img_last)
    ax1=fig.add_subplot(2,2,3)
    ims=[]
    for time in xrange(np.shape(clustering)[0]):
        im = ax1.imshow(clustering[time,:,:],cmap='gray')
        ims.append([im])
    #run animation
    ani = anim.ArtistAnimation(fig,ims, interval=100,blit=False)
    plt.title('predicted clustering trajectory')
    fig.add_subplot(2,2,2)
    img = Image.fromarray(y*255)
    plt.imshow(img)
    plt.title('ground truth clustering')
    fig.add_subplot(2,2,4)
    #plt.imshow(x)
    plt.plot(cost_log)
    plt.title('cost trajectory')
    last = clustering[-1]
    #print 'Norm of difference:',np.linalg.norm(last-y)/(y.shape[0])**2
    print 'Norm of difference:',np.linalg.norm(last-y)
    #ani.save('./animation.gif',writer='imagemagick',fps=30)
    #fig.save('/tmp/fig.gif', writer='imagemagick', fps=30)
    
    plt.show()
    #pdb.set_trace()
def run2():
    global embedder,clusterer,tf_clustering,data_params,k
    '''
    n,d = 2*1,3
    k = 2
    data_params = n,d
    v1 = np.array([0,0,0])[np.newaxis,:]
    x1 = np.tile(v1,[n/2,1])
    v2 = np.array([1,0,0])[np.newaxis,:]
    x2 = np.tile(v2,[n/2,1])
    x = np.vstack((x1,x2))
    y1 = np.hstack((np.ones((1,n/2)),np.zeros((1,n/2))))
    y2 = np.hstack((np.zeros((1,n/2)),np.ones((1,n/2))))
    y = y1.T*y1+y2.T*y2
    '''
    n,d = 4*10,3
    k = 4
    data_params = 4*n,d
    np.random.seed(2018)
    v1 = np.array([0,0,0])[np.newaxis,:]
    #v1 = np.random.rand(1,3)
    x1 = np.tile(v1,[2*n+n/4,1])
    v2 = np.array([1,0,0])[np.newaxis,:]
    #v2 = np.random.rand(1,3)
    x2 = np.tile(v2,[n/4,1])
    v3 = np.array([0,1,0])[np.newaxis,:]
    #v3 = np.random.rand(1,3)
    x3 = np.tile(v3,[n/4,1])
    v4 = np.array([0,0,1])[np.newaxis,:]
    #v4 = np.random.rand(1,3)
    x4 = np.tile(v4,[n+n/4,1])
    
    y1 = np.hstack((np.ones((1,2*n+n/4)),np.zeros((1,n+3*n//4))))
    y2 = np.hstack((np.zeros((1,2*n+n/4)),np.ones((1,n/4))))
    y2 = np.hstack((y2,np.zeros((1,n+n/2))))
    y3 = np.hstack((np.zeros((1,2*n+n/2)),np.ones((1,n/4))))
    y3 = np.hstack((y3,np.zeros((1,n+n/4))))
    y4 = np.hstack((np.zeros((1,2*n+3*n/4)),np.ones((1,n+n/4))))
    y = y1*y1.T+y2*y2.T+y3*y3.T+y4*y4.T
    #pdb.set_trace()

    x1 = np.vstack((x1,x2))
    x2 = np.vstack((x3,x4))
    x = np.vstack((x1,x2))
    #perm = np.random.permutation(np.eye(4*n))
    #x = np.matmul(perm,x)
    #y = np.matmul(perm,np.matmul(y,perm.T))
    ''
    #clusterer = EMClusterer(data_params,k)
    clusterer = GDKMeansClusterer2(data_params,k)
    tf_clustering = clusterer.infer_clustering()
    tf_membership = clusterer.history_list
    feed_dict = {clusterer.x:x}
    [clustering_history,cost_history,membership_history,grad_log] = sess.run([tf_clustering,clusterer.cost_log,clusterer.history_list,clusterer.grad_log],feed_dict=feed_dict)
    #[clustering_history,membership_history] = sess.run([tf_clustering,tf_membership],feed_dict=feed_dict) # for em
    #cost_history = []
    plotup(clustering_history,x,y,cost_history)
    last_cluster = clustering_history[-1]
    last_membership = membership_history[-1]

    pdb.set_trace()
def run3():
    # test the new gdkmeans clusterer
    n,d = 10,3
    data_params = n,d
    k = 4
    x = np.random.rand(n,d)
    clusterer = GDKMeansClusterer2(data_params,k)
    tf_clustering = clusterer.infer_clustering()
    feed_dict = {clusterer.x:x}
    sess.run(tf_clustering,feed_dict=feed_dict)
print('Starting TF Session')
sess = tf.InteractiveSession()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
run2()
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

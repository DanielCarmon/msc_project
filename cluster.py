import tensorflow as tf
import numpy as np
import pdb
from tqdm import tqdm


class BaseClusterer():
    def infer_clustering(self):
        for _ in tqdm(range(self.n_iters), desc="Building {} Layers".format(self.__class__.__name__)):
            self.update_params()
        history_tensor = tf.convert_to_tensor(self.history_list)  # [n_iters,n,k] tensor of membership matrices
        # history_tensor = tf.Print(history_tensor,[0],"Finished Parameter Updates")
        self.pred = self.get_clustering_matrices(history_tensor)  # [n_iters,n,n] tensor of similarity matrices
        self.pred = tf.identity(self.pred, name="PredictionHistory")
        return self.pred

    @staticmethod
    def get_clustering_matrices(history_tensor):
        # args:
        #   - history_tensor: [n_iters,n,k] tensor, who's [i,:,:]th element is a membership matrix inferred at i'th step of inner-optimization process.
        # output:
        #   - ys: [n_iters,n,n] tensor, who's [i,:,:]th element is a clustering matrix corresponding to bs[i,:,:]
        ys = tf.einsum('tij,tkj->tik', history_tensor, history_tensor)  # thanks einstein
        # check = tf.is_nan(tf.reduce_sum(ys))
        return ys


class GDKMeansClusterer1(BaseClusterer):
    ''' optimize over membership logits '''

    def __init__(self, data_params, k, n_iters=50):
        self.learn_rate = 10.
        self.curr_step = 0
        self.n_iters = n_iters
        self.k = k
        self.n, self.d = tuple(data_params)
        self.init_params()  # TF constants for inner-optimization
        self.grad_log = []
        self.cost_log = []

    def set_data(self, x):
        self.x = x

    def init_params(self):
        self.b = tf.random_normal([self.n, self.k], seed=2018)  # membership matrix
        # self.b = tf.constant(np.float32([[10,-10],[-10,10]])) # Optimizing over membership logits
        # self.b = tf.zeros([self.n,self.k])
        # debug init:
        '''
        n = 4*10
        y1 = np.hstack((np.ones((1,2*n+n/4)),np.zeros((1,n+3*n//4))))
        y2 = np.hstack((np.zeros((1,2*n+n/4)),np.ones((1,n/4))))
        y2 = np.hstack((y2,np.zeros((1,n+n/2))))
        y3 = np.hstack((np.zeros((1,2*n+n/2)),np.ones((1,n/4))))
        y3 = np.hstack((y3,np.zeros((1,n+n/4))))
        y4 = np.hstack((np.zeros((1,2*n+3*n/4)),np.ones((1,n+n/4))))
        y1 = np.hstack((y1.T,y2.T))
        y2 = np.hstack((y3.T,y4.T))
        y = np.hstack((y1,y2))
        self.b = tf.constant(y)
        self.b = tf.cast(self.b,tf.float32)
        '''
        self.b = tf.Print(self.b, [self.b[0], self.b[1]], "Init Logits:")
        self.history_list = []  # different self.b's across optimization

    def update_params(self):  # overrides super class method
        self.curr_step += 1
        b_max = tf.reduce_mean(self.b, axis=1)[:, tf.newaxis]
        self.b = self.b - b_max
        self.b = tf.Print(self.b, [self.b[0, 0], self.b[0, 1], self.b[0, 2], self.b[0, 3]], "Before Update:")

        cost = self.obj_f(self.b, self.x)
        self.cost_log.append(cost)
        grads = tf.gradients(cost, self.b)[0]
        grads = tf.Print(grads, [cost], "cost:")
        grads = tf.Print(grads, [tf.reduce_sum(grads ** 2)], "grads_total:")
        self.grad_log += [grads]
        # mu = self.learn_rate / np.sqrt(self.curr_step)
        mu = self.learn_rate
        self.b = self.b - mu * grads  # update
        self.b = tf.Print(self.b, [self.b[0, 0], self.b[0, 1], self.b[0, 2], self.b[0, 3]], "After Update:")
        self.b = tf.Print(self.b, [""], "--------{}--------".format(str(self.curr_step)))
        b_log = tf.nn.softmax(self.b, dim=1)  # from logits to distributions over clusters
        self.history_list.append(b_log)  # log

    @staticmethod
    def obj_f(b, x):
        b_probs = tf.nn.softmax(b, dim=1)
        centroid_matrix = GDKMeansClusterer1.get_centroid_matrix(b_probs, x)
        for i in range(4):
            centroid_matrix = tf.Print(centroid_matrix, [centroid_matrix[i]], "centroid_{}:".format(str(i)))
        ret = tf.norm(tf.matmul(b_probs, centroid_matrix) - x) ** 2
        return ret

    @staticmethod
    def get_centroid_matrix(b, x):
        # returns [NUM_CLUSTERS,EMBED_DIM] tf tensor
        # x = tf.Print(x,[x[0],x[-1],"|",b[0],b[1]],"x,b = ")
        #for i in range(1):
        #    x = tf.Print(x, [b[i]], "b_probs[{}]:".format(str(i)))
        clst_sz = tf.reduce_sum(b, axis=0)  # == tf.transpose(b)*tf.ones([n,1])
        #clst_sz = tf.Print(clst_sz, [clst_sz], message='cluster sizes:')
        inv_sz = tf.matrix_inverse(tf.diag(clst_sz))
        matmul_tmp = tf.matmul(inv_sz, tf.transpose(b))
        centroid_matrix = tf.matmul(matmul_tmp, x)
        check = tf.is_nan(tf.reduce_sum(centroid_matrix))
        # centroid_matrix = tf.Print(centroid_matrix,[check],message="check no. 98")
        return centroid_matrix


class GDKMeansClusterer2(BaseClusterer):
    ''' optimize over cluster centroids '''

    def __init__(self, data_params, k, learn_rate, n_iters=50, planted_values = False):
        self.learn_rate = learn_rate
        self.curr_step = 0
        self.n_iters = n_iters
        self.k = k
        GDKMeansClusterer2.k = k  # todo: fix this
        self.n, self.d = tuple(data_params) 
        self.planted_values = planted_values
        self.init_params()  # TF constants for inner-optimization
        self.grad_log = []
        self.cost_log = []
        self.diff_history = []
        self.maxgrad_history = []
    def set_data(self, x):
        self.x = x
    def init_params(self):
        if self.planted_values:
            print 'setting up planted values based centroid matrix'
            self.b_ph = tf.placeholder(tf.float32,[self.n,self.k]) # belief placeholder
            self.c = GDKMeansClusterer1.get_centroid_matrix(self.b_ph,self.x)
        else:
            self.c = tf.random_normal([self.k, self.d], seed=2018)  # centroid matrix
        # self.c = self.x[0],self.x[90],self.x[105],self.x[-1]
        # self.c = tf.Print(self.c,[self.c[i] for i in range(4)],"Init centroids:")
        self.history_list = []  # different membership matrices across optimization

    def update_params(self):  # overrides super class method
        self.curr_step += 1
        
        cost = self.obj_f(self.c, self.x)
        #cost = tf.Print(cost, [cost, self.x], 'cost')
        #cost = tf.constant(1.) * cost
        self.cost_log.append(cost)
        # cost = nan_alarm(cost)
        grads = tf.gradients(cost, self.c)[0]
        # grads = nan_alarm(grads)
        self.grad_log += [grads]
        maxgrad = tf.reduce_max(grads)
        self.maxgrad_history.append(maxgrad)
        #mu = self.learn_rate / np.sqrt(self.curr_step)
        mu = self.learn_rate
        # self.c = tf.Print(self.c,[self.c],"before")
        old = self.c
        self.c = self.c - mu * grads  # update
        diff = tf.reduce_sum((old-self.c)**2)
        self.diff_history.append(diff)
        # self.c = tf.Print(self.c,[self.c],"after")
        # self.c = tf.Print(self.c,[""],"--------{}--------".format(str(self.curr_step)))
        ""
        self.history_list.append(self.get_membership_matrix(self.c, self.x))  # log
    
    @staticmethod
    def obj_f(c, x):
        membership_matrix = GDKMeansClusterer2.get_membership_matrix(c, x)
        corresponding_c = tf.matmul(membership_matrix, c)
        ret = tf.reduce_sum((corresponding_c - x) ** 2)
        return ret

    @staticmethod
    def get_membership_matrix(c, x):
        # returns [n,k] tensor
        k = GDKMeansClusterer2.k
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(c), name='outer_subtraction')  # [n,d,k]
        distance_mat = tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k]
        # distance_mat = tf.Print(distance_mat,[tf.shape(distance_mat),distance_mat],"dist_mat:",summarize=30)
        inv_tmp = 1  # control softmax sharpness
        membership_mat = tf.nn.softmax(inv_tmp * (-distance_mat), 1)
        '''
        # non-differentiable:
        argmins = tf.argmin(distance_mat,axis=1)
        #argmins = tf.Print(argmins,[argmins],'argmins')
        membership_mat = tf.one_hot(argmins,k)
        '''
        '''
        for i in range(300):
            membership_mat = tf.Print(membership_mat,[membership_mat[i][j] for j in range(3)],"beliefs{}:".format(str(i)))
        '''
        return membership_mat


class KMeansClusterer(BaseClusterer):
    ''' Classic Hard-decision clustering '''

    def __init__(self, data_params, k, n_iters=100):
        self.curr_step = 0
        self.n_iters = n_iters
        self.k = k
        self.n, self.d = tuple(data_params)
        self.init_params()  # TF constants for inner-optimization
        self.history_list = []
        self.diff_history = []
    def set_data(self, x):
        self.x = x

    def init_params(self):
        self.c = 1e-1* tf.random_normal([self.k, self.d], seed=2018)  # centroid matrix

    def update_params(self):  # overrides super class method
        self.curr_step += 1

        self.old_c = self.c
        self.c = self.clean_empty_centroids()
        self.b = self.get_membership_matrix(self.c,self.x, self.k)

        self.c = self.get_centroid_matrix(self.b,self.x, self.k)
        diff = tf.reduce_sum((self.c-self.old_c)**2)
        self.diff_history.append(diff)
        self.history_list.append(self.b)
    def infer_clustering(self):
        for _ in tqdm(range(self.n_iters), desc="Building {} Layers".format(self.__class__.__name__)):
            self.update_params()
        history_tensor = tf.convert_to_tensor(self.history_list)  # [n_iters,n,k] tensor of membership matrices
        return history_tensor
    def clean_empty_centroids(self):
        self.membership_mat = self.get_membership_matrix(self.c, self.x, self.k)
        cond = self.has_empty_centroid
        body = self.replace_centroid
        self.replace_mask = tf.zeros((self.k, self.d))  # init
        ##self.c = tf.Print(self.c,[self.c],"Start cleaning loop")
        wl = tf.while_loop(cond, body, [self.c])
        ##wl = tf.Print(wl,[0],"End cleaning loop")
        return wl
    def replace_centroid(self, c):  # cleaning loop body
        k, d = self.k, self.d
        replace_mask = self.replace_mask
        rand_mat = tf.random_uniform((k, d))/1000
        ##rand_mat = tf.Print(rand_mat,[rand_mat],"replacing empty centroid with:")
        add = replace_mask * rand_mat
        sub = replace_mask * c
        return c + add - sub
    def has_empty_centroid(self, c):  # cleaning loop cond
        x = self.x
        self.membership_mat = self.get_membership_matrix(c, x, self.k)
        self.cluster_sums = tf.reduce_sum(self.membership_mat, axis=0)
        ths = tf.constant(1.)
        clust_sums_clipped = tf.clip_by_value(self.cluster_sums, 0, ths)
        avg = tf.reduce_mean(clust_sums_clipped)
        bool_val = avg < ths  # True iff one of the entries is below ths
        replace_indicator = tf.one_hot(tf.argmin(clust_sums_clipped), self.k)[tf.newaxis, :]
        replace_mask = tf.transpose(tf.tile(replace_indicator, [self.d, 1]))
        self.replace_mask = replace_mask
        # bool_val = tf.Print(bool_val,[bool_val],"has empty centroid?")
        return bool_val
    @staticmethod
    def get_membership_matrix(c, x, k):
        # returns [n,k] tensor
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(c), name='outer_subtraction')  # [n,d,k]
        distance_mat = tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k]
        argmins = tf.argmin(distance_mat,axis=1)
        #argmins = tf.Print(argmins,[argmins],'argmins')
        membership_mat = tf.one_hot(argmins,k)
        return membership_mat
    @staticmethod
    def get_centroid_matrix(b, x, k):
        # returns [NUM_CLUSTERS,EMBED_DIM] tf tensor
        ##for i in range(1):
        ##    x = tf.Print(x, [b[i]], "b_probs[{}]:".format(str(i)))
        clst_sz = tf.reduce_sum(b, axis=0)  # == tf.transpose(b)*tf.ones([n,1])
        ##clst_sz = tf.Print(clst_sz, [clst_sz[i] for i in range(k)], message='cluster sizes:')
        inv_sz = tf.matrix_inverse(tf.diag(clst_sz))
        matmul_tmp = tf.matmul(inv_sz, tf.transpose(b))
        centroid_matrix = tf.matmul(matmul_tmp, x)
        check = tf.is_nan(tf.reduce_sum(centroid_matrix))
        # centroid_matrix = tf.Print(centroid_matrix,[check],message="check no. 98")
        return centroid_matrix


class EMClusterer(BaseClusterer):
    def __init__(self, data_params, k, bandwidth = 0.5, n_iters=20):
        self.n_iters = n_iters
        self.bandwidth = bandwidth
        self.k = k
        self.n, self.d = tuple(data_params)
        # self.x = tf.placeholder(tf.float32,[self.n,self.d]) # rows are data points
        self.init_params()  # TF constants for inner-optimization

    def set_data(self, x):
        #x = tf.Print(x,[x],'input x:')
        self.x = x

    def init_params(self):
        self.theta = tf.random_normal([self.k, self.d], seed=2017, name='theta_0')
        # self.theta = tf.constant(np.float32([[0.,1],[1.,0.]]))
        self.history_list = []
        self.diff_history = []
    def update_params(self):
        self.z = self.infer_z(self.x, self.theta, self.bandwidth)
        old_theta = self.theta
        self.theta = self.infer_theta(self.x, self.z)  # update
        diff = tf.reduce_sum((old_theta-self.theta)**2)
        ##diff = tf.Print(diff,[diff],"diff:")
        self.diff_history.append(diff)
        # self.theta = tf.Print(self.theta,[z],"Z:")
        # self.theta = tf.Print(self.theta,[selftheta],"Theta:")
        self.history_list.append(self.z)  # log

    @staticmethod
    def infer_theta(x, z):
        #x = tf.Print(x,[x[0],x[1],"|",z[0],z[1]],"Entered infer_theta with x,z = ")
        clust_sums = tf.matmul(tf.transpose(z), x, name='clust_sums')  # [k,d]
        #clust_sums = tf.Print(clust_sums, [clust_sums], "clust_sums",summarize=100)
        clust_sz = tf.reduce_sum(z, axis=0, name='clust_sz')  # [k]
        eps = 1e-1
        clust_sz+=eps
        #clust_sz = tf.Print(clust_sz, [clust_sz], "clust_sz",summarize=100)
        normalizer = tf.matrix_inverse(tf.diag(clust_sz), name='normalizer')  # [k,k]
        # normalizer = tf.Print(normalizer,[normalizer[0],normalizer[1]],"normalizer:")
        theta = tf.matmul(normalizer, clust_sums)  # [k,d] soft centroids
        # theta = tf.Print(theta,[theta[0],theta[1]],"inferred Theta:")
        return theta
    @staticmethod
    def infer_z(x, theta, bandwidth):
        #x = tf.Print(x,[x[0],x[1],"|",theta[0],theta[1]],"Entered infer_z func with x,theta = ")
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(theta), name='out_sub')  # [n,d,k]
        z = -tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k]
        # numerically stable calculation:
        z = z - tf.reduce_mean(z, axis=1)[:, None]
        z = tf.nn.softmax(bandwidth*z, dim=1)
        check = tf.is_nan(tf.reduce_sum(z))
        #z = tf.Print(z,[z[0],z[1],check],"inferred Z:")
        return z

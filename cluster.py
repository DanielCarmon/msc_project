import tensorflow as tf
import numpy as np
import pdb
from tqdm import tqdm

class BaseClusterer():
    def infer_clustering(self):
        for _ in tqdm(range(self.n_iters), desc="Building {} Layers".format(self.__class__.__name__)):
            self.update_params()
        ys_assign_history = tf.convert_to_tensor(self.history_list)  # [n_iters,n,k] tensor of membership matrices
        return ys_assign_history

    @staticmethod
    def get_clustering_matrices(history_tensor):
        # args:
        #   - history_tensor: [n_iters,n,k] tensor, who's [i,:,:]th element is a membership matrix inferred at i'th step of inner-optimization process.
        # output:
        #   - ys: [n_iters,n,n] tensor, who's [i,:,:]th element is a clustering matrix corresponding to bs[i,:,:]
        ys = tf.einsum('tij,tkj->tik', history_tensor, history_tensor)  # thanks einstein
        # check = tf.is_nan(tf.reduce_sum(ys))
        return ys


class GDKMeansPlusPlus(BaseClusterer):
    ''' Optimize over cluster centroids.
        Initialize centroids as in Kmeans++
        Use Gumbel-Softmax trick to approximate a sampling from discrete dist.
    '''

    def __init__(self, data_params, k, learn_rate, n_iters=50, planted_values = False):
        self.learn_rate = learn_rate
        self.curr_step = 0
        self.n_iters = n_iters
        self.k = k
        self.n, self.d = tuple(data_params) 
        self.planted_values = planted_values
        self.grad_log = []
        self.cost_log = []
        self.diff_history = []
        self.maxgrad_history = []
    def set_data(self, x):
        '''
            Sets x as input.
            In addition, builds initialization pipeline for centroids
        '''
        self.x = x
        self.init_params()
    def init_params(self):
        data = self.x
        old_centroids = self.init_first_centroid(data)
        for i in range(self.k):
            old_centroids = self.centroid_choice_layer(data,old_centroids)
        self.c = old_centroids
        self.history_list = []  # different membership matrices across optimization

    @staticmethod
    def get_dist_mat(a, b):
        reduce_norms = lambda a: tf.reduce_sum((a * a), axis=1)
        global norms
        norms1 = reduce_norms(a)[:, None]
        norms2 = reduce_norms(b)[None, :]
        norms = b
        # norms = tf.concat([norms1,tf.transpose(norms2)],0)
        global inner_prod_mat
        inner_prod_mat = tf.matmul(a, tf.transpose(b))
        dist_mat = (-2 * inner_prod_mat + norms1) + norms2
        return dist_mat

    @staticmethod
    def get_prob_vector(data, old_centroids):
        '''
        input:
            data - [n,d] data matrix
            old_centroids - [k',d] centroid matrix
        output:
            [n,1] vector proportional to D^2(x_i) = softmin(d(xi,c1),...,d(xi,ck'))
        '''
        global dist_mat
        dist_mat = GDKMeansPlusPlus.get_dist_mat(data, old_centroids)
        means = tf.reduce_mean(dist_mat, 1)
        dist_mat_normed = dist_mat - means[:, None]
        bandwidth = 0.05
        softmin_mat = tf.nn.softmax(-bandwidth * dist_mat_normed, axis=1)  # softmin
        D = softmin_mat * dist_mat  # elementwise
        D = tf.reduce_sum(D, 1)
        D = D / tf.reduce_sum(D)
        return D

    @staticmethod
    def centroid_choice_layer(data, old_centroids):
        global prob_vector, new_centroid_coeffs
        prob_vector = GDKMeansPlusPlus.get_prob_vector(data,
                                      old_centroids)  # [n]. normalized (i.e a probability inducing) vector of soft-min distances from old centroids
        eps = 1e-6
        prob_vector += eps
        prob_vector /= tf.reduce_sum(prob_vector)
        relaxedOneHot = tf.contrib.distributions.RelaxedOneHotCategorical
        dist = relaxedOneHot(temperature=.05, probs=prob_vector)
        new_centroid_coeffs = dist.sample()  # differentiable op. shape [n]
        new_centroid = tf.matmul(new_centroid_coeffs[None, :], data)
        old_centroids = tf.concat([old_centroids, new_centroid], 0)
        return old_centroids

    @staticmethod
    def init_first_centroid(data):
        old_centroids = data[0, :][None, :]  # no need to permute since feeded data is already permuted
        return old_centroids

    def update_params(self):  # overrides super class method
        self.curr_step += 1
        cost = self.obj_f(self.c, self.x)
        self.cost_log.append(cost)
        grads = tf.gradients(cost, self.c)[0]
        self.grad_log += [grads]
        maxgrad = tf.reduce_max(grads)
        self.maxgrad_history.append(maxgrad)
        mu = self.learn_rate
        old = self.c
        self.c = self.c - mu * grads  # update
        diff = tf.reduce_sum((old-self.c)**2)
        self.diff_history.append(diff)
        ""
        self.history_list.append(self.get_membership_matrix(self.c, self.x))  # log
    
    @staticmethod
    def obj_f(c, x):
        membership_matrix = get_membership_matrix(c, x)
        corresponding_c = tf.matmul(membership_matrix, c)
        ret = tf.reduce_sum((corresponding_c - x) ** 2)
        return ret

    @staticmethod
    def get_membership_matrix(c, x):
        # returns [n,k] tensor
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(c), name='outer_subtraction')  # [n,d,k]
        distance_mat = tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k]
        # distance_mat = tf.Print(distance_mat,[tf.shape(distance_mat),distance_mat],"dist_mat:",summarize=30)
        inv_tmp = 1  # control softmax sharpness
        membership_mat = tf.nn.softmax(inv_tmp * (-distance_mat), 1)
        return membership_mat


class EMClusterer(BaseClusterer):
    def __init__(self, data_params, k, bandwidth = 0.5, n_iters=20,init=0):
        self.n_iters = n_iters
        self.bandwidth = bandwidth
        self.init = init
        self.k = k # no. clusters
        self.n, self.d = tuple(data_params) # n = batch size, d = data dim

    def set_data(self, x):
        #x = tf.Print(x,[x],'input x:')
        self.x = x
        self.init_params()  # TF constants for inner-optimization. Init might be data-driven

    @staticmethod
    def get_dist_mat(a, b):
        reduce_norms = lambda a: tf.reduce_sum((a * a), axis=1)
        global norms
        norms1 = reduce_norms(a)[:, None]
        norms2 = reduce_norms(b)[None, :]
        norms = b
        # norms = tf.concat([norms1,tf.transpose(norms2)],0)
        global inner_prod_mat
        inner_prod_mat = tf.matmul(a, tf.transpose(b))
        dist_mat = (-2 * inner_prod_mat + norms1) + norms2
        return dist_mat

    @staticmethod
    def get_prob_vector(data, old_centroids):
        '''
        input:
            data - [n,d] data matrix
            old_centroids - [k',d] centroid matrix
        output:
            [n,1] vector proportional to D^2(x_i) = softmin(d(xi,c1),...,d(xi,ck'))
        '''
        global dist_mat
        dist_mat = EMClusterer.get_dist_mat(data, old_centroids)
        means = tf.reduce_mean(dist_mat, 1)
        dist_mat_normed = dist_mat - means[:, None]
        bandwidth = 0.05
        softmin_mat = tf.nn.softmax(-bandwidth * dist_mat_normed, axis=1)  # softmin
        D = softmin_mat * dist_mat  # elementwise
        D = tf.reduce_sum(D, 1)
        D = D / tf.reduce_sum(D)
        return D

    @staticmethod
    def centroid_choice_layer(data, old_centroids):
        global prob_vector, new_centroid_coeffs
        prob_vector = EMClusterer.get_prob_vector(data,
                                      old_centroids)  # [n]. normalized (i.e a probability inducing) vector of soft-min distances from old centroids
        eps = 1e-6
        prob_vector += eps
        prob_vector /= tf.reduce_sum(prob_vector)
        relaxedOneHot = tf.contrib.distributions.RelaxedOneHotCategorical
        dist = relaxedOneHot(temperature=.05, probs=prob_vector)
        new_centroid_coeffs = dist.sample()  # differentiable op. shape [n]
        new_centroid = tf.matmul(new_centroid_coeffs[None, :], data)
        old_centroids = tf.concat([old_centroids, new_centroid], 0)
        return old_centroids

    @staticmethod
    def init_first_centroid(data):
        old_centroids = data[0, :][None, :]  # no need to permute since feeded data is already permuted
        return old_centroids
    def init_params(self):
        '''
        cases of self.init:
        0: Random init
        1: Fixed-Random init. Same random init every call. randomness sampled at compile time
        2: Soft-Kmeans++ init. 
        '''
        if self.init==0:
            print 'using init 0'
            self.theta = tf.random_normal([self.k, self.d], seed=2018, name='theta_0') # seed fixed randomness at sess level. every sess run this gives a new value
        elif self.init==1:    
            print 'using init 1'
            np.random.seed(2018)
            rand_init = np.random.normal(size=[self.k,self.d])
            self.theta = tf.constant(rand_init,name = 'theta_0')
            self.theta = tf.cast(self.theta,tf.float32)
        elif self.init==2:
            print 'using init 2'
            data = self.x
            old_centroids = EMClusterer.init_first_centroid(data)
            for i in range(self.k-1):
                old_centroids = EMClusterer.centroid_choice_layer(data,old_centroids)
            self.theta = old_centroids

        self.history_list = []
        self.diff_history = []
    def update_params(self):
        self.z = self.infer_z(self.x, self.theta, self.bandwidth)
        old_theta = self.theta
        self.theta = self.infer_theta(self.x, self.z)  # update
        diff = tf.reduce_sum((old_theta-self.theta)**2)
        self.diff_history.append(diff)
        self.history_list.append(self.z)  # log

    @staticmethod
    def infer_theta(x, z):
        clust_sums = tf.matmul(tf.transpose(z), x, name='clust_sums')  # [k,d]
        clust_sz = tf.reduce_sum(z, axis=0, name='clust_sz')  # [k]
        eps = 1e-1
        clust_sz+=eps
        normalizer = tf.matrix_inverse(tf.diag(clust_sz), name='normalizer')  # [k,k]
        theta = tf.matmul(normalizer, clust_sums)  # [k,d] soft centroids
        return theta
    @staticmethod
    def infer_z(x, theta, bandwidth):
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(theta), name='out_sub')  # [n,d,k]
        z = -tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k]
        # numerically stable calculation:
        z = z - tf.reduce_mean(z, axis=1)[:, None]
        z = tf.nn.softmax(bandwidth*z, axis=1)
        #check = tf.is_nan(tf.reduce_sum(z))
        #z = tf.Print(z,[z[0],z[1],check],"inferred Z:")
        return z
if __name__ == '__main__':
    ## testing km++ init
    n,d = 100,1001
    k = 10
    np_data = np.zeros((0,d))
    for i in range(k):
        mu_i = np.zeros((1,d))
        mu_i[0,i]=100
        cluster_data = np.random.normal(loc=mu_i,size=(n/k,d))
        np_data = np.vstack((np_data,cluster_data))
    # computation graph:
    data = tf.placeholder(tf.float32,[n,d])
    clusterer = EMClusterer([n,d],k,init=2)
    clusterer.set_data(data)
    sess = tf.InteractiveSession()
    feed_dict = {data:np_data}
    np_theta = sess.run(clusterer.theta,feed_dict)
    print np_theta
    i_in = []
    for row in np_theta:
        print np.argmax(row),np.max(row)
        i_in.append(np.argmax(row))
    i_in = set(i_in)
    print 'out:',k-len(i_in)

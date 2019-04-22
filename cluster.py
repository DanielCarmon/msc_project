import tensorflow as tf
import numpy as np
import pdb
from tqdm import tqdm

class BaseClusterer():
    def infer_clustering(self):
        for _ in tqdm(range(self.n_iters), desc="Building {} Layers".format(self.__class__.__name__)):
            self.update_params()
        print 'before:',self.history_list
        ys_assign_history = tf.convert_to_tensor(self.history_list)  # [n_iters,n,k] tensor of membership matrices
        print 'after:',ys_assign_history
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
        bw = 0.05
        softmin_mat = tf.nn.softmax(-bw * dist_mat_normed, axis=1)  # softmin
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
    def __init__(self, data_params, k, bw_params, n_iters=20,init=0,infer_covar=False):
        self.n_iters = n_iters
        self.em_bw, self.gumbel_temp, self.softmin_bw = bw_params
        self.init = init
        self.k = k # no. clusters
        self.n, self.d = tuple(data_params) # n = batch size, d = data dim
        self.infer_covar = infer_covar
        if self.infer_covar:
            # init params
            self.alpha = tf.constant(np.ones(k),tf.float32)
    def set_data(self, x):
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
    def get_prob_vector(data, old_centroids,softmin_bw):
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
        softmin_mat = tf.nn.softmax(-softmin_bw * dist_mat_normed, axis=1)  # softmin
        D = softmin_mat * dist_mat  # elementwise
        D = tf.reduce_sum(D, 1)
        D = D / tf.reduce_sum(D)
        return D

    @staticmethod
    def centroid_choice_layer(data, old_centroids,gumbel_temp,softmin_bw):
        global prob_vector, new_centroid_coeffs
        prob_vector = EMClusterer.get_prob_vector(data,old_centroids,softmin_bw)  # [n]. normalized (i.e a probability inducing) vector of soft-min distances from old centroids
        eps = 1e-3
        prob_vector += eps
        prob_vector /= tf.reduce_sum(prob_vector)
        relaxedOneHot = tf.contrib.distributions.RelaxedOneHotCategorical
        dist = relaxedOneHot(temperature=gumbel_temp, probs=prob_vector)
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
                old_centroids = EMClusterer.centroid_choice_layer(data,old_centroids,self.gumbel_temp,self.softmin_bw)
            self.theta = old_centroids
        if self.infer_covar:
            ## init sigmas as covariance matrices for clusters formed by
            ## assigning each initial-cluster-mean with points for whom
            ## he minimizes distance among all initial-cluster-means
            dist_mat = EMClusterer.get_dist_mat(self.x, self.theta) # distance matrix of data from clusters
            prebeliefs = tf.nn.softmax(dist_mat) # assign each data point a closest-to cluster mean
            self.initial_sigma,self.cache = EMClusterer.infer_sigma(self.x,prebeliefs,self.theta) # [k,d,d], intra-cluster covariances
            self.initial_sigma += tf.convert_to_tensor([tf.eye(self.d) for i in range(self.k)])
            self.sigma = self.initial_sigma
            # init alphas:
            n = tf.shape(prebeliefs)[0]
            self.alpha = tf.reduce_sum(prebeliefs,axis=0)/tf.cast(n,tf.float32)
            self.alpha = tf.Print(self.alpha,[self.alpha],"alpha:",summarize=10)

        self.history_list = []
        self.diff_history = []
    def update_params(self):
        if self.infer_covar:
            for tens in [self.alpha,self.theta,self.sigma,self.x]:
                print 'preE:',tens
            self.z = EMClusterer.Estep(self.alpha,self.theta,self.sigma,self.x)
            old_theta = self.theta
            for tens in [self.alpha,self.theta,self.sigma,self.z]:
                print 'preM:',tens
            self.alpha,self.theta,self.sigma= EMClusterer.Mstep(self.x,self.z)
            for tens in [self.alpha,self.theta,self.sigma]:
                print 'postM:',tens
        else:
            self.z = self.infer_z(self.x, self.theta, self.em_bw) # E-step
            old_theta = self.theta
            self.theta = self.infer_theta(self.x, self.z)  # M-step
        diff = tf.reduce_sum((old_theta-self.theta)**2)
        print diff
        self.diff_history.append(diff)
        print 'updated beliefs:',self.z
        self.history_list.append(self.z)  # log

    @staticmethod
    def Estep(alpha,mu,sigma,X):
        n = tf.shape(X)[0]
        k = tf.shape(mu)[0]
        d = tf.shape(X)[1]
        def soften_probs(probs,eps=1e-3):
            probs = probs+eps
            probs = probs/tf.reduce_sum(probs)
            return probs
        #sigma = tf.Print(sigma,[sigma[0]],message="sigma0:",summarize=100)
        inv_sigma = tf.linalg.inv(sigma) # [k,d,d]
        #inv_sigma = tf.Print(inv_sigma,[inv_sigma],message="inv_sigma:",summarize=100)
        out_sub = tf.subtract(X[:, :, None], tf.transpose(mu), name='out_sub')  # [n,d,k]
        out_sub = tf.transpose(out_sub,[0,2,1])[...,tf.newaxis] # [n,k,d,1]
        broadcasted_covar = tf.broadcast_to(inv_sigma,[n,k,d,d]) # [n,k,d,d]
        matmul1 = tf.matmul(broadcasted_covar,out_sub) # [k,n,d,1]
        matmul2 = tf.matmul(out_sub,matmul1,transpose_a=True) # [n,k,1,1]
        exponent = -0.5*(tf.squeeze(matmul2,[2,3])) # [n,k]
        #sigma = tf.Print(sigma,[tf.is_nan(tf.reduce_sum(sigma))],message="Before:")
        dets = tf.linalg.det(sigma) # [k]
        #dets = tf.Print(dets,[tf.is_nan(tf.reduce_sum(dets))],message="After:")
        #dets = tf.Print(dets,[dets],"dets:",summarize=100)
        denom = tf.sqrt(dets)
        eps = 1e-1
        probs1 = tf.exp(exponent)/(denom+eps) # [n,k]
        probs1 = probs1/tf.reduce_sum(probs1)
        probs2 = soften_probs(probs1)
        likelihood = tf.reduce_sum(probs2*alpha,axis=1) # [n]
        Znew = probs2*alpha/likelihood[:,tf.newaxis] # [n,k]
        for tens in [matmul2,exponent,dets,denom,probs1,probs2,likelihood,Znew]:
            print 'in Estep:',tens,tens.shape
        return Znew
    @staticmethod
    def Mstep(X,Z):
        n = tf.cast(tf.shape(X)[0],tf.float32)
        c_weights = tf.reduce_sum(Z,0)
        alpha = c_weights/n
        mu = tf.matmul(X,Z,transpose_a=True)/n
        mu = tf.transpose(mu) # [k,d]
        sigma,_ = EMClusterer.infer_sigma(X,Z,mu)
        for tens in [alpha,mu,sigma]:
            print 'at M Step:', tens,tens.shape
        return alpha,mu,sigma
    @staticmethod
    def infer_theta(x, z):
        """ Infer Gaussian means """
        clust_sums = tf.matmul(tf.transpose(z), x, name='clust_sums')  # [k,d]
        clust_sz = tf.reduce_sum(z, axis=0, name='clust_sz')  # [k]
        eps = 1e-1
        clust_sz+=eps
        normalizer = tf.matrix_inverse(tf.diag(clust_sz), name='normalizer')  # [k,k]
        theta = tf.matmul(normalizer, clust_sums)  # [k,d] soft centroids
        return theta
    @staticmethod
    def infer_sigma(X,Z,mu):
        """
             Infer covariance matrices of Gaussians
             X: [n,d] data matrix
             Z: [n,k] belief matrix
             mu: [k,d] cluster means
        """
        c_weights = tf.reduce_sum(Z,0) # points per cluster
        n = tf.shape(Z)[0]
        k = tf.shape(Z)[1]
        d = tf.shape(X)[1]
        out_sub = tf.subtract(X[:, :, None], tf.transpose(mu), name='out_sub')  # [n,d,k]
        out_sub = tf.transpose(out_sub,[2,0,1]) # [k,n,d] tensor of diff vectors between X,mu
        tmp1,tmp2 = out_sub[:,:,:,tf.newaxis],out_sub[:,:,tf.newaxis,:]
        expanded = tf.matmul(tmp1,tmp2) # [k,n,d,d] tensor of dyad per diff vector
        broadcasted = tf.broadcast_to(tf.transpose(Z),(d,d,k,n)) #
        broadcasted = tf.transpose(broadcasted,[2,3,0,1]) # [k,n,d,d]
        weighted = expanded*broadcasted # [k,n,d,d] belief-weighted dyads
        summed = tf.reduce_sum(weighted,axis=1) # [k,d,d] sum of belief-weighted dyads
        normed = tf.broadcast_to((1./c_weights),[d,d,k]) # [d,d,k]
        sigma = tf.transpose(normed,[2,0,1])*summed # [k,d,d]
        return sigma,locals()
    @staticmethod
    def infer_z(x, theta, bw):
        """ Infer belief matrices """
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(theta), name='out_sub')  # [n,d,k]
        z = -tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k] distance matrix
        # numerically stable calculation:
        z = z - tf.reduce_mean(z, axis=1)[:, None]
        z = tf.nn.softmax(bw*z, axis=1) # when using inferred variance, need different one per cluster/column
        check = tf.is_nan(tf.reduce_sum(z))
        #z = tf.Print(z,[z[0],z[1],check],"inferred Z:")
        return z
    @staticmethod
    def infer_z2(x, theta, bw):
        """ Infer belief matrices """
        outer_subtraction = tf.subtract(x[:, :, None], tf.transpose(theta), name='out_sub')  # [n,d,k]
        z = -tf.reduce_sum(outer_subtraction ** 2, axis=1)  # [n,k] distance matrix
        # numerically stable calculation:
        z = z - tf.reduce_mean(z, axis=1)[:, None]
        z = tf.nn.softmax(bw*z, axis=1) # when using inferred variance, need different one per cluster/column
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

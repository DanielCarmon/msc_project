import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from functools import reduce
from embed import *
from cluster import *

def nan_alarm(x):
    return tf.Print(x, [tf.is_nan(tf.reduce_sum(x))], "{} Nan?".format(x.name))

def get_clustering_matrices(y_assign_history):
    # args:
    #   - y_assign_history: [n_iters,n,k] tensor, who's [i,:,:]th element is a membership matrix inferred at i'th step of inner-optimization process.
    # output:
    #   - y_partition_history: [n_iters,n,n] tensor, who's [i,:,:]th element is a clustering matrix corresponding to bs[i,:,:]
    y_partition_history = tf.einsum('tij,tkj->tik', y_assign_history, y_assign_history)  # thanks einstein
    return y_partition_history

def my_entropy(p_vals):
    ''' args: [k,] shaped tensor with positive entries that sum to 1. '''
    nats = tf.log(p_vals)
    logs = nats/tf.log(2.)
    return -tf.reduce_sum(p_vals*logs)

def my_nmi(y_assign_gt,y_assign_predict):
    ''' args: two row-stochastic matrices of shape [n,k]'''
    n = tf.reduce_sum(y_assign_gt) # row-stochastic with n rows...
    # get probabilities from assignments:
    eps = 1e-3
    joint_prob_mat = tf.matmul(y_assign_gt,y_assign_predict,transpose_a=True)/n # [k,k]
    p_vals1 = tf.reduce_sum(y_assign_gt,axis=0) + eps
    p_vals1 = p_vals1/tf.reduce_sum(p_vals1) # normalize
    p_vals2 = tf.reduce_sum(y_assign_predict,axis=0) + eps
    p_vals2 = p_vals2/tf.reduce_sum(p_vals2) # normalize
    p_vals_joint = tf.reshape(joint_prob_mat,[-1]) + eps
    p_vals_joint = p_vals_joint/tf.reduce_sum(p_vals_joint) # normalize
    # calculate entropies:
    entropy1 = my_entropy(p_vals1)
    entropy1 = tf.Print(entropy1,[entropy1],'entropy_gt:')
    entropy2 = my_entropy(p_vals2)
    entropy2 = tf.Print(entropy2,[entropy2],'entropy_pred:')
    entropy_joint = my_entropy(p_vals_joint)
    entropy_joint = tf.Print(entropy_joint,[entropy_joint],'entropy_joint:')
    # use mutual info identity:
    mutual_info = entropy1+entropy2-entropy_joint
    # return normalized: 
    normalizer = tf.sqrt(entropy1*entropy2)
    normalizer = tf.Print(normalizer,[normalizer],"normalizer: ")
    normalizer *= 1.
    return mutual_info/normalizer # sqrt is problematic. change to arithmetic avg?


class Model:
    optimizer_class = tf.train.AdamOptimizer
    def __init__(self, data_params, embedder=None, clusterer=None, lr = 0.0001, is_img=False, sess=None, for_training=False, regularize=True, use_tg=True,obj='L2',log_grads=False):
        self.sess = sess
        self.embedder = embedder
        self.clusterer = clusterer
        self.n, self.d = tuple(data_params)
        if is_img:
            self.x = tf.placeholder(tf.float32, [None, self.d, self.d, 3])  # rows are data points
        else:
            self.x = tf.placeholder(tf.float32, [self.n, self.d])  # rows are data points
        self.x = tf.cast(self.x, tf.float32)
        #self.y = tf.placeholder(tf.float32, [self.n, self.n])
        self.y = tf.placeholder(tf.float32, [None, None]) #[n,k]
        self.y = tf.cast(self.y, tf.float32)
        self.x_embed = self.embedder.embed(self.x,for_training) # embeddings tensor
        ##self.x_embed = tf.Print(self.x_embed, [self.x_embed], "x_embed:", summarize=10)
        self.lr = lr
        self.optimizer = self.optimizer_class(lr)
        self.clusterer.set_data(self.x_embed) # compose clusterer on top of embedder
        self.clustering = self.clusterer.infer_clustering() # get output clustering
        # compose loss func on top of output clustering:
        if obj=='L2':
            self.loss = self.L2_loss(self.clustering, self.y, use_tg)
        elif obj=='nmi':
            self.loss = self.NMI_loss(self.clustering, self.y, use_tg)
        else:
            print 'Error: Unsupported objective "',obj,'"'
            exit()
        if log_grads:
            self.pre_grads = tf.gradients(self.loss, embedder.params) 
            self.grads = filter((lambda x: x!=None),self.pre_grads) # remove None gradients from tf's batch norm params.
            self.loss = tf.Print(self.loss,self.grads , 'gradient:')
        else: 
            self.grads = tf.constant(-1)
            self.loss = tf.Print(self.loss,[self.loss], 'loss:')
        #self.loss = tf.Print(self.loss, [self.loss], 'loss:')
        
        regularizer,beta = 0.,0.
        if regularize:
            for param in self.embedder.params:
                print 'regularizing ',param
                regularizer += tf.nn.l2_loss(param)
        self.loss = beta*regularizer + self.loss
        print 'building optimizer'
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.loss)
        print 'initializing global variables'
        self.sess.run(tf.global_variables_initializer())
        self.embedder.load_weights(self.sess)
    @staticmethod
    def L2_loss(y_pred, y, use_tg):
        '''
        args:
            y_pred: [n_iters,n,k] tensor. history of soft assignments
            y: [n,k] tensor. ground truth assignment
            use_tg: bool. whether use trajectory ("aux") gradients or not
        '''
        y_pred = get_clustering_matrices(y_pred)
        compare = y_pred[-1] # no trajectory gradient
        if use_tg: # trajectory gradient
            compare = y_pred
        tensor_shape = tf.shape(compare)
        normalize = tf.reduce_prod(tensor_shape) # num of entries
        # normalize = 1.
        # print 'gradient normalizing factor = ',normalize
        y = tf.matmul(y,y,transpose_b=True)
        return tf.reduce_sum((compare - y) ** 2) / tf.cast(normalize, tf.float32)
    @staticmethod
    def NMI_loss(y_pred,y,use_tg):
        print 'in NMI_loss'
        compare = y_pred[-1]# no support for tg yet
        return -my_nmi(y,compare) # optimizer should minimize this

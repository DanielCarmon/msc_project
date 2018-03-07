import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nn import *
import pdb
from tqdm import tqdm
from embed import *
from cluster import *


def nan_alarm(x):
    return tf.Print(x, [tf.is_nan(tf.reduce_sum(x))], "{} Nan?".format(x.name))


class Model:
    optimizer_class = tf.train.AdamOptimizer
    #optimizer_class = tf.train.GradientDescentOptimizer
    def __init__(self, data_params, embedder=None, clusterer=None, lr = 0.0001, is_img=False, sess=None, for_training=False, regularize=True):
        self.sess = sess
        self.embedder = embedder
        self.clusterer = clusterer
        self.n, self.d = tuple(data_params)
        if is_img:
            #self.x = tf.placeholder(tf.float32, [self.n, self.d, self.d, 3])  # rows are data points
            self.x = tf.placeholder(tf.float32, [None, self.d, self.d, 3])  # rows are data points
        else:
            self.x = tf.placeholder(tf.float32, [self.n, self.d])  # rows are data points
        self.x = tf.cast(self.x, tf.float32)
        #self.y = tf.placeholder(tf.float32, [self.n, self.n])
        self.y = tf.placeholder(tf.float32, [None, None])
        self.y = tf.cast(self.y, tf.float32)
        self.for_training = for_training
        self.x_embed = self.embedder.embed(self.x, self.for_training)
        ##self.x_embed = tf.Print(self.x_embed, [self.x_embed], "x_embed:", summarize=10)
        self.lr = lr
        self.optimizer = self.optimizer_class(lr)
        self.clusterer.set_data(self.x_embed)
        self.clustering = self.clusterer.infer_clustering()

        self.loss = self.loss_func(self.clustering, self.y)
        # self.loss = self.stam(self.x_embed,self.y)
        self.grads = tf.gradients(self.loss, embedder.params)  # gradient
        self.grads = filter((lambda x: x!=None),self.grads) # remove None gradients from tf's batch norm params.
        ##self.loss = tf.Print(self.loss, [self.loss], 'loss:')
        #self.loss = tf.Print(self.loss, [self.grads],'grad')
        #for i in range(1):
        #    self.loss = tf.Print(self.loss,[self.grads[0]],'grad{}'.format(str(i)))
        self.loss = tf.Print(self.loss, [tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in self.grads])  ], 'gradient:', summarize=100)

        regularizer,beta = 0.,1e-10
        if regularize:
            for param in self.embedder.params:
                print 'regularizing ',param
                regularizer += tf.nn.l2_loss(param)
        self.loss = beta*regularizer + self.loss
        self.loss = 1.*self.loss
        self.train_step = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.embedder.load_weights(self.sess)
    @staticmethod
    def loss_func(y_pred, y):
        # y = tf.Print(y,[tf.shape(y),tf.shape(y_pred),y,y_pred],"y:",summarize=100)
        compare = y_pred[-1]
        tensor_shape = tf.shape(compare)
        normalize = (tensor_shape[0] * tensor_shape[1])
        # normalize = 1.
        # print 'gradient normalizing factor = ',normalize
        return tf.reduce_sum((compare - y) ** 2) / tf.cast(normalize, tf.float32)

    @staticmethod
    def stam(x, y):
        return tf.reduce_sum(x)+tf.reduce_sum(y)

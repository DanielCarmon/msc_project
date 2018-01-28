import tensorflow as tf
import numpy as np
from nn import *

class BaseEmbedder():
    def __init__(self):
        pass

    def embed(self):
        pass


class ImgEmbedder(BaseEmbedder):
    def __init__(self, data_params):
        self.img_dim, self.c = tuple(data_params)
        self.x = tf.placeholder(tf.float32, [self.img_dim, self.img_dim, 3])
        self.x_new = self.embed()

    """
    def embed(self):
        # encode pixel coordinates
        img_dim,num_col = self.img_dim,self.c
        x = self.x
        inds = np.arange(0,img_dim)
        prefix0 = np.tile(inds,[img_dim])
        prefix1 = np.repeat(inds,[img_dim]*img_dim)
        prefix = np.vstack((prefix0,prefix1)).T
        tf_prefix = tf.constant(prefix) # [img_dim**2,2]
        tf_prefix = tf.cast(tf_prefix,tf.float32)
        tf_prefix = 0*tf_prefix/(img_dim**2)
        x_embed = tf.reshape(x,[img_dim**2,num_col]) # [img_dim**2,num_col]
        x_embed = tf.concat([tf_prefix,x_embed],axis=1) #concat x,y coords to d
        x_embed = tf.Print(x_embed,[x_embed])
        x_embed = tf.identity(x_embed,name='x_embed')
        return x_embed # [n,5]
    """

    def embed(self):
        img_dim, num_col = self.img_dim, self.c
        x = self.x
        x_embed = tf.reshape(x, [img_dim ** 2, num_col])  # [img_dim**2,num_col]
        x_embed = tf.Print(x_embed, [x_embed])
        x_embed = tf.identity(x_embed, name='x_embed')
        return x_embed


class DeepSetEmbedder1(BaseEmbedder):
    'embedding flat vectors'
    hidden_layer_width = [3]

    def __init__(self, data_params, embed_dim=3):
        self.img_dim, self.c = tuple(data_params)
        self.embed_dim = embed_dim
        self.init_network()

    def init_network(self):
        self.hidden_layer_width = [self.img_dim] + self.hidden_layer_width + [self.embed_dim]
        self.rho1 = MLP([1])
        self.rho2 = MLP([1])
        self.phi1 = MLP([1])
        self.phi2 = MLP([1])

    def embed(self):
        return NotImplemented


class ProjectionEmbedder(BaseEmbedder):
    def __init__(self, data_params):
        self.n, self.d = tuple(data_params)
        self.init_params()  # TF trainable Variable
        self.param_history = []
        # self.x_new = self.embed()

    def init_params(self):
        self.params = tf.get_variable("embedding_matrix", [self.d, 1])
        # self.params = tf.Variable(np.ones((3,1)))
        # self.params = tf.cast(self.params,tf.float32)

    def embed(self, x):
        w = self.params
        # w = tf.Print(w,[w],"params:")
        ret = tf.matmul(x, w)
        return ret

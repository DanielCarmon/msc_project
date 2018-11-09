import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
from nets import nets_factory
import sys
import pdb
from scipy.misc import imread, imresize
slim = tf.contrib.slim

def tf_print(x,msg=''):
    x = tf.Print(x,[x],msg)
    x = 1.*x
    return x

class MLP():

    def __init__(self,layer_widths,x=None,name='',initializer = tf.ones_initializer()):
        # args:
        # n_hidden: list of layer widths
        # x: TF tensor
        self.name = name
        self.out = x # this will be changed at build. self.out <- next_layer(self.out)
        self.layer_widths = layer_widths
        self.activations = [x]
        self.params = []
        self.initializer = initializer
        self.build()
    
    def build(self):
        activation = tf.nn.relu
        initializer = self.initializer
        depth = len(self.layer_widths)
        for i in range(depth-1):
            #initializer = tf.contrib.layers.xavier_initializer()
            weight_matrix = tf.get_variable("{}_PermCovarWeightMatrix{}".format(self.name,str(i)), [self.layer_widths[i],self.layer_widths[i+1]],initializer=initializer)
            self.params.append(weight_matrix)
            weight_matrix = tf.Print(weight_matrix,[weight_matrix],'weight_matrix:')
            #weight_matrix = tf.Variable(np.eye(self.layer_widths[0])) #@debug
            #weight_matrix = tf.constant(np.eye(self.layer_widths[0])) #@debug
            weight_matrix = tf.cast(weight_matrix,tf.float32)
            self.activations.append(self.out)
            self.out = tf.matmul(self.out,weight_matrix)
            #self.out = activation(self.out)
    
    def output(self):
        return self.out


class BaseEmbedder:
    
    def __init__(self):
        self.params = []

    def compose(self, other):
        """ return lambda x: self(other(x))"""
        ret = BaseEmbedder()
        def new_embed(x):
            inner_embed = other.embed(x)
            other.out = inner_embed
            self.out = self.embed(inner_embed)
            ret.params = self.params + other.params # self: PermCovarEmbedder, other: InceptionEmbedder, ret: BaseEmbedder
            # the permcovar embedder's params are constructed only at the line 'self.embed(inner_embed)'
            return self.out
        ret.permcovar_pointer = self
        ret.embed = new_embed
        ret.params = self.params + other.params
        try:
            ret.load_weights = other.load_weights
            ret.weight_file = other.weight_file
            ret.sess = other.sess
            ret.pretrained = other.pretrained
            ret.built = other.built
            ret.saver = other.saver
            ret.params = other.params
            ret.endpoints = other.endpoints
        except:
            print 'problem'
            exit()
            pass
        return ret


class PermCovarEmbedder1(BaseEmbedder):

    hidden_layer_width = [3]

    def __init__(self, input_dim):
        self.d = input_dim
        self.params = []
    '''
    def embed(self, x):
        layer_widths = [self.d,self.d]
        self.mlp = MLP(layer_widths,x)
        out = self.mlp.output()
        #return x
        return out
    '''
    def embed(self,x):
        x = tf_print(x,msg='Embedding input')
        layer_widths = [self.d,self.d] # or something else
        eps = 1e-3
        ##init = tf.initializers.constant(value=eps)
        init = tf.contrib.layers.xavier_initializer()
        self.phi = MLP(layer_widths,x,"phi",init)
        phi_matrix = self.phi.output() # [n,d'] matrix of phi1(x_i)s.
        phi_matrix = tf_print(phi_matrix,msg='phi matrix')
        phi_sum = tf.reduce_sum(phi_matrix,0) # [d]
        phi_sums_matrix = phi_sum-phi_matrix # [n,d']
        layer_widths = [self.d,self.d] # or something else
        self.rho = MLP(layer_widths,phi_sums_matrix,"rho",init)
        context_matrix = self.rho.output()
        context_matrix = tf_print(context_matrix,msg='context matrix')
        layer_widths = [2*self.d,self.d] # or something else
        concat_matrix = tf.concat([x,context_matrix],1)
        ##init = tf.initializers.identity()
        self.psi = MLP(layer_widths,concat_matrix,"psi",init)
        out = self.psi.output()
        out = tf_print(out,msg='Embedding output')
        self.params += (self.phi.params+self.rho.params+self.psi.params)
        return out

class InceptionEmbedder(BaseEmbedder):
    def __init__(self, weight_file=None, sess=None, output_layer='Logits',new_layer_width=-1,weight_decay=4e-5):
        self.weight_file = weight_file
        self.sess = sess
        self.pretrained = self.built = False
        self.new_layer_width = new_layer_width
        self.saver = None
        self.params = []
        self.output_layer = output_layer
        self.endpoints = 0
        self.inception_dim = 1001
        self.weight_decay = weight_decay

    def embed(self, x, is_training = False):
        network_fn = nets_factory.get_network_fn( # define network
            'inception_v3',
            num_classes=self.inception_dim,
            weight_decay=self.weight_decay,
            is_training=is_training)
        self.logits,self.activations_dict = network_fn(x) # build network
        self.params = tf.trainable_variables()
        self.param_dict = dict([(var.name,var) for var in self.params])
        output_layer_name = self.output_layer
        try:
            self.output = self.activations_dict[output_layer_name]; print "using '{}' as output layer".format(output_layer_name)
        except:
            print 'Error. unknown layer name:',output_layer_name
            print 'Please use one of these names:'
            print self.activations_dict.keys()
            exit()
        if self.new_layer_width!=-1: #
            print 'building new layer'
            initializer = tf.contrib.layers.xavier_initializer()
            self.new_layer_w = tf.get_variable('new_layer_w',[self.inception_dim,self.new_layer_width],initializer=initializer)
            self.output = tf.matmul(self.output,self.new_layer_w)
        else:
            self.output = self.logits
        return self.output

    def load_weights(self,sess):
        print 'start loading pre-trained weights'
        vars_to_restore = sess.graph.get_collection('variables')
        self.batch_norm_vars = list(set(vars_to_restore).difference(set(self.params)))
        self.batch_norm_vars = list(filter(lambda v: 'new_layer' not in v.name,self.batch_norm_vars))
        restorer = slim.assign_from_checkpoint_fn(
              self.weight_file,
              vars_to_restore,
              ignore_missing_vars=True)
        restorer(sess)
        print 'finished loading pre-trained weights'

        #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    def save_weights(self,sess):
        save_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/model.ckpt"
        if self.saver is None:
            self.saver = tf.train.Saver(self.param_dict)
        print 'checkpoint saved at:', self.saver.save(sess,save_path)
        return save_path

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    weight_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-v3/inception_v3.ckpt" # params pre-trained on ImageNet
    embedder = InceptionEmbedder(weight_path,new_layer_width=100)
    np.random.seed(137)
    const = np.random.random((100,299,299,3))
    x = tf.constant(const)
    x = tf.cast(x,tf.float32)
    x_embed = embedder.embed(x,is_training=False)
    sess.run(tf.global_variables_initializer())
    embedder.load_weights(sess)
    print sess.run(x_embed)[-1][-1]
    print sess.run(x_embed)[-1][-1]
    np_activations = sess.run((embedder.activations_dict).values())
    f = open('act2','w+')
    for i in range(len(np_activations)):
        f.write(str(embedder.activations_dict.keys()[i])+'--->'+str(np.mean(np_activations[i])))
        f.write('\n')
    f.close()
    pdb.set_trace()

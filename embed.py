import tensorflow as tf
import numpy as np
import pickle
from nn import *
from tqdm import tqdm
import sys
sys.path.insert(0,'/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/inception-inference')
import inception_model as inception
from scipy.misc import imread, imresize
import pdb

def tf_print(x,msg=''):
    x = tf.Print(x,[x],msg)
    x = 1.*x
    return x

def my_batch_norm(batch_x):
    batch_mean = tf.reduce_mean(batch_x, 0)
    batch_var = tf.reduce_mean((batch_x-batch_mean)**2, 0)
    eps = 0.0001
    return (batch_x-batch_mean)/tf.sqrt(batch_var)

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
            weight_matrix = tf.get_variable("{}_DeepSetWeightMatrix{}".format(self.name,str(i)), [self.layer_widths[i],self.layer_widths[i+1]],initializer=initializer)
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
            ret.params = self.params + other.params # self: DeepSetEmbedder, other: InceptionEmbedder, ret: BaseEmbedder
            # the deepset embedder's params are constructed only at the line 'self.embed(inner_embed)'
            return self.out
        ret.deepset_pointer = self
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

class DeepSetEmbedder1(BaseEmbedder):
    """embedding flat vectors"""
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
    ''


class ProjectionEmbedder(BaseEmbedder):
    def __init__(self, data_params):
        self.n, self.d = tuple(data_params)
        self.init_params()  # TF trainable Variable
        self.param_history = []
        # self.x_new = self.embed()

    def init_params(self):
        self.params = [tf.get_variable("embedding_matrix", [self.d, 1])]
        # self.params = tf.Variable(np.ones((3,1)))
        # self.params = tf.cast(self.params,tf.float32)

    def embed(self, x):
        w = self.params[0]
        # w = tf.Print(w,[w],"params:")
        ret = tf.matmul(x, w)
        return ret


class Vgg16Embedder(BaseEmbedder):
    def __init__(self, weights=None, sess=None, embed_dim = 512):
        self.embed_dim = embed_dim
        self.weight_file = weights
        self.sess = sess
        self.pretrained = self.built = False
        self.params = []
    def convlayers(self, x):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            caltech_birds_mean=[123.90631071, 127.40118913, 110.10152148]
            imagenet_means = [123.68, 116.779, 103.939]
            mean = tf.constant(caltech_birds_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = x - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv1_1 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv1_2 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv2_1 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv2_2 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv3_1 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv3_2 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv3_3 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv4_1 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv4_2 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv4_3 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv5_1 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv5_2 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_norm = my_batch_norm(out)
            self.conv5_3 = tf.nn.relu(out_norm, name=scope)
            self.params += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            fc1l_norm = my_batch_norm(fc1l)
            self.fc1 = tf.nn.relu(fc1l_norm)
            self.params += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            fc2l_norm = my_batch_norm(fc2l)
            self.fc2 = tf.nn.relu(fc2l_norm)
            self.params += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.params += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print 'started loading pre-trained params'
        for i, k in enumerate(keys):
            # print i, k, np.shape(weights[k])
            sess.run(self.params[i].assign(weights[k]))
        print 'finished loading pre-trained params'

    def embed(self, x):
        if not self.built:
            self.convlayers(x)
            self.fc_layers()
            self.fc3l = tf.Print(self.fc3l,[self.fc3l],message="self.fc3l:",summarize=20)
            self.fc3l_norm = my_batch_norm(self.fc3l)
            self.output = self.fc3l_norm
            self.built = True
        if not self.pretrained:
            if self.weight_file is not None and self.sess is not None:
                self.load_weights(self.weight_file, self.sess)
            self.pretrained = True
        return self.output

class InceptionEmbedder(BaseEmbedder):
    def __init__(self, weight_file=None, sess=None, output_layer='logits',new_layer_width=-1):
        self.weight_file = weight_file
        self.sess = sess
        self.pretrained = self.built = False
        self.new_layer_width = new_layer_width
        self.saver = None
        self.params = []
        self.output_layer = output_layer
        self.endpoints = 0
        self.inception_dim = 1001
    def embed(self, x, for_training = False):
        print x
        self.logits,self.activations_dict = inception.inference(x,self.inception_dim,for_training=for_training)
        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore() # dictionary
        self.param_dict = variables_to_restore
        self.params = [param for param in self.param_dict.values()]
        output_layer_name = self.output_layer
        try:
            self.output = self.activations_dict[output_layer_name]; print "using '{}' as output layer".format(output_layer_name)
        except:
            print 'Error. unknown layer name:',output_layer_name
            print 'Please use one of these names:'
            print self.activations_dict.keys()
            exit()
        if self.new_layer_width!=-1: #
            initializer = tf.contrib.layers.xavier_initializer()
            self.new_layer_w = tf.get_variable('new_layer_w',[self.inception_dim,self.new_layer_width],initializer=initializer)
            self.output = tf.matmul(self.output,self.new_layer_w)
        return self.output
    def load_weights(self,sess):
        print 'start loading pre-trained weights'
        ckpt = tf.train.get_checkpoint_state(self.weight_file)  
        self.saver = tf.train.Saver(self.param_dict)
        self.saver.restore(sess, ckpt.model_checkpoint_path)  
        #self.params = filter((lambda x: x!=None),self.params)

        print 'finished loading pre-trained weights'

        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    def save_weights(self,sess):
        save_path = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/model.ckpt"
        if self.saver is None:
            self.saver = tf.train.Saver(self.param_dict)
        print 'checkpoint saved at:', self.saver.save(sess,save_path)
        return save_path

'''
    def infer(img_path):
        img = imread(img_path, mode='RGB')
        img = imresize(img, (224, 224))
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print class_names[p], prob[p]
'''
'''
    def infer_batch(dir_path):
        import glob
        filenames = glob.glob(dir_path + '/*.jpg')
        img_batch = []
        for fname in filenames[:5]:
            img = imread(fname, mode='RGB')
            img = imresize(img, (224, 224))
            # img = img[np.newaxis,:,:,:]
            img = img.astype(np.float64)
            img_batch.append(img)
            print fname
        print 'done with fnames'
        img_batch = np.array(img_batch)
        print "let's feed"
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: img_batch})
        return prob
'''
'''
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    path = 'laska.png'
    infer(path)
'''

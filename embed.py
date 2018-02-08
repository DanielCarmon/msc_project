import tensorflow as tf
import numpy as np
from nn import *
from tqdm import tqdm
from scipy.misc import imread, imresize
import pdb


class BaseEmbedder:
    def __init__(self):
        self.params = []

    def embed(self, x):
        pass

    def compose(self, other):
        """ return lambda x: self(other(x))"""
        ret = BaseEmbedder()
        ret.embed = lambda x: self.embed(other.embed(x))
        ret.params = self.params + other.params
        return ret


class TestEmbedder(BaseEmbedder):
    def __init__(self, data_params):
        self.n, self.img_dim = data_params
        self.params = [tf.get_variable("embedding_matrix", [3 * (self.img_dim) ** 2, 1])]

    def embed(self, x):
        img_dim, num_col = self.img_dim, 3
        x_embed = tf.reshape(x, [self.n, (img_dim ** 2) * num_col])  # [img_dim**2,num_col]
        w = self.params[0]
        # w = tf.Print(w,[w],"params:")
        ret = tf.matmul(x_embed, w)
        ret = ret / tf.norm(ret, 1)
        return ret


class TestEmbedder2(BaseEmbedder):
    def __init__(self, data_params):
        self.n, self.d = data_params
        self.params = []

    def embed(self, x):
        mat = tf.Variable(np.random.rand(self.d, self.d))
        mat = tf.cast(mat, tf.float32)
        self.params.append(mat)
        return tf.matmul(x, mat)


class ImgEmbedder(BaseEmbedder):
    """
    Embeds a batch of n [d,d,3] images to n [1,embed_dim] vectors
    """

    def __init__(self, data_params):
        self.n, self.img_dim = data_params
        self.params = []

    def embed(self, x):
        img_dim, num_col = self.img_dim, 3
        x_embed = tf.reshape(x, [self.n, (img_dim ** 2) * num_col])  # [img_dim**2,num_col]
        x_embed = tf.Print(x_embed, [x_embed])
        x_embed = tf.identity(x_embed, name='x_embed')
        return x_embed

    """
    def embed(self, x):
        # encode pixel coordinates
        pdb.set_trace()
        img_dim = self.img_dim
        inds = np.arange(0, img_dim)
        prefix0 = np.tile(inds, [img_dim])
        prefix1 = np.repeat(inds, [img_dim] * img_dim)
        prefix = np.vstack((prefix0, prefix1)).T
        tf_prefix = tf.constant(prefix)  # [img_dim**2,2]
        tf_prefix = tf.cast(tf_prefix, tf.float32)
        # lambda = 0.1 # hyper-parameter for importance of 2d closeness
        # tf_prefix = lambda * tf_prefix / (img_dim ** 2)
        tf_prefix_tiled = tf.tile(tf_prefix[tf.newaxis, :], [self.n, 1,])
        x_embed = tf.reshape(x, [self.n, (img_dim ** 2)*3])  # [self.n, img_dim**2 * num_col]
        x_embed = tf.concat([tf_prefix_tiled, x_embed], axis=1)  # concat x,y coords to d
        x_embed = tf.Print(x_embed, [x_embed])
        x_embed = tf.identity(x_embed, name='x_embed')
        return x_embed  # [self.n, 5]

    """


class DeepSetEmbedder1(BaseEmbedder):
    """embedding flat vectors"""
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

    def embed(self, x):
        return NotImplemented


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
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = x - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
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
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
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
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
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
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
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
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.params += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
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
            self.fc1 = tf.nn.relu(fc1l)
            self.params += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
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
            self.probs = tf.nn.softmax(self.fc3l)
            self.probs = tf.Print(self.probs,[self.probs],message="probs:",summarize=20)
            self.mylayer = tf.Variable(tf.truncated_normal([1000, self.embed_dim], dtype=tf.float32, stddev=1e-1), name='mylayer')
            
            self.last = tf.matmul(self.probs,self.mylayer)
            self.output = tf.nn.relu(self.last)
            self.built = True
        if not self.pretrained:
            if self.weight_file is not None and self.sess is not None:
                self.load_weights(self.weight_file, self.sess)
            self.pretrained = True
        return self.output

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

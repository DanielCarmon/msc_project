import tensorflow as tf
import numpy as np


class NN():
    pass

class MLP(NN):
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

class vgg16:
    def __init__(self,retrain=False):
        pass
'''
# Single Hidden Layer#
n_hidden = [5, 50, 1]
mlp = MLP(n_hidden)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
n_train_steps = 50

bsz = 1
x, y = np.random.rand(bsz, 5), 0 * np.random.rand(bsz, 1)
feed_dict = {mlp.x: x, mlp.y: y}
for i in range(n_train_steps):
    sess.run(mlp.train_step, feed_dict=feed_dict)
    w = sess.run(mlp.layers, {mlp.x: x, mlp.y: y})
    print(i, sess.run(mlp.loss, feed_dict=feed_dict))
    print(w[0])
    print(w[1])
    print('-------------------------------------')
'''


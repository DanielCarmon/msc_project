import tensorflow as tf
import numpy as np

class NN():
    pass
class MLP(NN):
    lr = .1
    optimizer = tf.train.AdamOptimizer(lr)
    def __init__(self,n_hidden):
        self.x = tf.placeholder(tf.float32,[None,n_hidden[0]]) # rows are data points
        last = self.x
        self.layers = []
        for i in range(1,len(n_hidden)):
            mat = tf.Variable(tf.random_normal([n_hidden[i-1], n_hidden[i]], stddev=0.35),
                      name="weights{}".format(str(i)))
            #activation_f = tf.nn.relu
            activation_f = tf.nn.sigmoid
            activation = activation_f(tf.matmul(last,mat))
            last = activation
            self.layers.append(last)
        self.output = last
        self.y = tf.placeholder(tf.float32,[None,n_hidden[-1]])
        self.loss = tf.norm(self.output-self.y)**2 # squared distance loss
        self.loss = tf.Print(self.loss,[self.loss],"loss:")
        self.train_step = self.optimizer.minimize(self.loss)
'''
#Single Hidden Layer:
n_hidden = [5,100,1]
mlp = MLP(n_hidden)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
n_train_steps = 500

bsz = 1
x,y = np.random.rand(bsz,5),np.random.rand(bsz,1)
for i in range(n_train_steps):
    sess.run(mlp.train_step,feed_dict={mlp.x:x,mlp.y:y})
    print i,sess.run(mlp.loss,feed_dict={mlp.x:x,mlp.y:y})
'''

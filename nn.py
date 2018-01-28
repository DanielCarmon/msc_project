import tensorflow as tf
import numpy as np


class NN():
    pass

class MLP(NN):
    lr = .1
    optimizer = tf.train.AdamOptimizer(lr)

    def __init__(self, n_hidden, x=None):
        if x == None:
            self.x = tf.placeholder(tf.float32, [None, n_hidden[0]])  # rows are data points
        else:
            self.x = x
        last = self.x
        self.layers = []
        for i in range(1, len(n_hidden)):
            self.last_built = i
            mat = self.init_weights([n_hidden[i - 1], n_hidden[i]])
            activation_f = tf.nn.sigmoid
            activation = activation_f(tf.matmul(last, mat))
            last = activation
            self.layers.append(last)
        self.output = last
        self.y = tf.placeholder(tf.float32, [None, n_hidden[-1]])
        self.loss = tf.norm(self.output - self.y) ** 2  # squared distance loss
        self.loss = tf.Print(self.loss, [self.loss], "loss:")
        self.train_step = self.optimizer.minimize(self.loss)

    def init_weights(self, shape):
        const_init = tf.random_normal(shape)
        i = self.last_built
        var_init = tf.Variable(const_init, name="weights{}".format(str(i)))
        return var_init


''
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
''

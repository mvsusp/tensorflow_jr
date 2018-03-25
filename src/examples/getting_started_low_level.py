from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from src import tensorflow_jr as tf

x = tf.placeholder(dtype=tf.float32)

y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(name='W', initial_value=np.array([1.]))

b = tf.Variable(name='b', initial_value=2.)

y_hat = x * W + b

loss = tf.reduce_sum(tf.square(y - y_hat))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss, [W])

x_train = 3  # [1, 2, 3],

y_train = 4  # [2, 4, 6]

init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)

for i in xrange(10):
    curr_W, curr_b, curr_loss, _ = session.run([W, b, loss, train], {x: x_train, y: y_train})

    print('W: {}, b: {}, loss: {}'.format(curr_W, curr_b, curr_loss))

prediction = session.run([y_hat], {x: 10})

print(prediction)

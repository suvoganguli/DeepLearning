import tensorflow as tf

name = 'experiment'
with tf.variable_scope(name):
    x = tf.placeholder(tf.float32, [None, 3], name='x')


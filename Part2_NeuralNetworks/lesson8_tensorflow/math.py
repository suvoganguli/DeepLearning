# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session

xy = tf.divide(tf.cast(tf.constant(x),tf.float32), tf.cast(tf.constant(y),tf.float32))
z = tf.subtract(xy, tf.cast(tf.constant(1),tf.float32))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)

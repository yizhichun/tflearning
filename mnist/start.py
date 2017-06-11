import tensorflow as tf

h = tf.constant("Hello tf!")
sess = tf.Session()

print(sess.run(h))


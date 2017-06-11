from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)

h1_units = 300

#input 
with tf.name_scope("input"):
	x = tf.placeholder(tf.float32, [None, 784], name='images')
	variable_summaries(x)
	y_ = tf.placeholder(tf.float32, [None, 10], name='labels')
	variable_summaries(y_)
	keep_prob = tf.placeholder(tf.float32) #drop inactivate rate

#model parameters
with tf.name_scope('w1'):
	#w1 = tf.Variable(tf.zeros([784,h1_units]))
	w1 = tf.Variable(tf.truncated_normal([784, h1_units], stddev=0.1))
	variable_summaries(w1)
with tf.name_scope('b1'):
	b1 = tf.Variable(tf.zeros([h1_units]))
	variable_summaries(b1)

with tf.name_scope('w2'):
	w2 = tf.Variable(tf.zeros([h1_units,10]), name='ww2')
	#w2 = tf.Variable(tf.truncated_normal([h1_units, 10], stddev=0.1))
	variable_summaries(w2)
with tf.name_scope('b2'):
	b2 = tf.Variable(tf.zeros([10]), name='bb2')
	variable_summaries(b2)

#output
with tf.name_scope('layer2'):
	layer2 = tf.nn.relu(tf.matmul(x, w1) + b1)
	layer2_drop = tf.nn.dropout(layer2, keep_prob)
	#variable_summaries(layer2)
with tf.name_scope("output"):
	y =  tf.nn.softmax(tf.matmul(layer2_drop, w2) + b2, name='y')
	variable_summaries(y)

#loss function
with tf.name_scope("loss"):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]), name='ce')
	tf.summary.scalar('crossentropy',cross_entropy)

#optimizer
with tf.name_scope('train'):
	train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#evaluation
with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1),name='cor_pre')
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
	tf.summary.scalar('accuracy',accuracy)

sess = tf.InteractiveSession()

#merge all summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)
test_writer = tf.summary.FileWriter('./test', sess.graph)

tf.global_variables_initializer().run()

for i in range(1000):
	if i%100 == 0:
		summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_:mnist.test.labels, keep_prob:0.75})
		test_writer.add_summary(summary,i);
		print('Accuracy at step %s: %s' % (i, acc))
	batch_xs, batch_ys = mnist.train.next_batch(100)
	summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
	train_writer.add_summary(summary,i)



print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

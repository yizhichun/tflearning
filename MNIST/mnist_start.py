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
    tf.summary.histogram('histogram', var)


#input 
with tf.name_scope("input"):
	x = tf.placeholder(tf.float32, [None, 784], name='images')
	variable_summaries(x)
	y_ = tf.placeholder(tf.float32, [None, 10], name='labels')
	variable_summaries(y_)

#model parameters
with tf.name_scope('weights'):
	W = tf.Variable(tf.zeros([784,10]), name='W')
	variable_summaries(W)
with tf.name_scope('biases'):
	b = tf.Variable(tf.zeros([10]), name='b')
	variable_summaries(b)

#output
with tf.name_scope("output"):
	y =  tf.nn.softmax(tf.matmul(x,W) + b, name='y')
	variable_summaries(y)	

#loss function
with tf.name_scope("loss"):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]), name='ce')
	tf.summary.scalar('crossentropy',cross_entropy)

#optimizer
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

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
		summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_:mnist.test.labels})
		test_writer.add_summary(summary,i);
		print('Accuracy at step %s: %s' % (i, acc))	
	batch_xs, batch_ys = mnist.train.next_batch(100)
	summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs, y_:batch_ys})
	train_writer.add_summary(summary,i)



print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

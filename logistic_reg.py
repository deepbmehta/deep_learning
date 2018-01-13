import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([1,10]))

logits = tf.matmul(x,w)+b




cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits) )
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		epoch_loss = 0
		for _ in range(int(mnist.train.num_examples/batch_size)):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,cost],feed_dict = {x: batch_x, y: batch_y})
			epoch_loss+=c
		print('Epoch',epoch,'completed out of',training_epochs,'loss:',epoch_loss)
	correct = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct,'float'))
	print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))	
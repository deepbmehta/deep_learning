import tensorflow as tf
x1 = tf.constant(5)
x2 = tf.constant(5)
# re = x1*x2

re = tf.multiply(x1,x2)
print(re)
ad = tf.add(x1,x2)
correct = tf.equal(x1,x2)
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
# sess = tf.Session()
# print(sess.run(re))
# sess.close()

with tf.Session() as sess:
	print(sess.run(re))
	print(sess.run(ad))
	print(sess.run(correct))
	print(sess.run(accuracy))



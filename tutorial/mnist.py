import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#   x is a placeholder, which is a value that we'll input when we ask TensorFlow to run a computation
#   [None, 784] is 2-D tensor. None means that a dimension can be of any length.
x = tf.placeholder(tf.float32, [None, 784])

#   Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
#   It can be used and modified by the conputation.
#   We can get 10-dimensional vectors by multiply W by x
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#   y is output
y = tf.nn.softmax(tf.matmul(x, W) + b)

#   To implement cross-entropy we need to first add a new placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#   Implement the cross-entropy
#   reduce_sum adds the elements in the second dimension of y, due to the reduction_indicie=[1] (the first dimension of y is 'None')
#   tf.reduce_mean computes the mean over all the examples in the batch
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#   We ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()

#   We can now launch the model in a Session
sess = tf.Session()
sess.run(init)

#   We'll run the training step 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#   Evaluation the model
#   tf.argmax(y_,1) is the correct label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#   Above equation gives us a list of booleans. ex.[True, False, True, True]
#   We cast to floating point numbers and then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

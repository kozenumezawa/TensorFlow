from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

#Initialize weights with a small amount of noise
def weight_variable(shape, Name):
    with tf.name_scope(Name) as scope:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=Name)

#Initialize bias with a
def bias_variable(shape, Name):
    with tf.name_scope(Name) as scope:
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=Name)

#   Convolution and Pooling
def conv2d(x, W):
    with tf.name_scope('conv2d') as scope:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    with tf.name_scope('max_pool_2x2') as scope:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='output')

W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")    #   [width, height, input, filters]
b_conv1 = bias_variable([32], "b_conv1")

#   To apply the layer, we first reshape x to a 4d tensor
#   the second and third dimensions correspoinding to image width and height
#   Final dimension correspoinding to the number of color channels
x_image = tf.reshape(x, [-1, 28, 28, 1])


#   Antivation: activation function in Relu(Rectified Linear Unit)
with tf.name_scope ('activation1'):
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) #   Pooling: reduce dimensions

#   Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

with tf.name_scope ('activation2'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#   Dropout
with tf.name_scope ('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Reedout Layer
W_fc2 = weight_variable([1024, 10], "W_fc2")
b_fc2 = bias_variable([10], "b_fc2")
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#   Train and Evaluate the Model
def cross_entropy(y_, y_conv):
    with tf.name_scope('error') as scope:
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                      tf.log(y_conv), reduction_indices=[1]))
        return cross_entropy

cross_entropy = cross_entropy(y_, y_conv)

def trainning(cross_entropy):
    with tf.name_scope('trainning') as scope:
        return tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = trainning(cross_entropy)

def accuracy(y_conv, y_):
    with tf.name_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

accuracy = accuracy(y_conv, y_)

tf.scalar_summary("x-entropy", cross_entropy)

#   Write placeholders in TensorBoard
summary_writer = tf.train.SummaryWriter('tensorboard-mnist_data', graph_def=sess.graph_def)
summary_op = tf.merge_all_summaries()

sess.run(tf.initialize_all_variables())

for i in range(300):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

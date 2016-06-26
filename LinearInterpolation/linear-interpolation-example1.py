#   This example uses not deep learning but newral-network

import tensorflow as tf

#   coordinates
input = [[0., 1., 1., 2., 2., 3.]]

answer_data = [[ 1., 0.]]

#   The inference() function return output
def inference(input_placeholder):
    with tf.name_scope('inference') as scope:
        W = tf.Variable(tf.zeros([6, 2]), name="weight") # W = [dimension(6), dimension(2)]
        b = tf.Variable(tf.zeros([2]), name="bias")      # b = [dimenstion(2)]

        y = tf.nn.softmax(tf.matmul(input_placeholder, W) + b)
    return y

def loss(output, supervisor_labels_placeholder):
    with tf.name_scope('loss') as scope:
        cross_entropy = -tf.reduce_sum(supervisor_labels_placeholder * tf.log(output))
        tf.scalar_summary("x-entropy", cross_entropy)
    return cross_entropy

def training(loss):
    with tf.name_scope('training') as scope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return train_step

#   Variables are defined here to write variables in TensorBoard
with tf.Graph().as_default():
    #   [None, 2] is 2-D tensor. None means that a dimension can be of any length.
    input_placeholder = tf.placeholder("float", [None, 6], name="input_placeholder")

    supervisor_labels_placeholder = tf.placeholder("float", [None,2], name="supervisor_labels_placeholder")


    #   feed_dict denotes input of placeholder
    feed_dict = {input_placeholder: input, supervisor_labels_placeholder: answer_data}

    #   calculate y = Wx + b
    output = inference(input_placeholder)

    loss = loss(output, supervisor_labels_placeholder)
    training_op = training(loss)

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        #   Write placeholders in TensorBoard
        summary_writer = tf.train.SummaryWriter('linear-interpolation_data', graph_def=sess.graph_def)
        sess.run(init)

        for step in range(1000):
            sess.run(training_op, feed_dict=feed_dict)
            if step % 100 == 0:
                print sess.run(loss, feed_dict=feed_dict)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

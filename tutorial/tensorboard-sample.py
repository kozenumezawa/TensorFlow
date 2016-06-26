#Reference: http://qiita.com/sergeant-wizard/items/c98597b8add04b8eea0b
#           http://qiita.com/sergeant-wizard/items/af7a3bd90e786ec982d2

import tensorflow as tf

input = [
[1., 0., 0.],
[0., 1., 0.],
[0., 0., 1.]
]

winning_hands = [
[0., 1., 0.],
[0., 0., 1.],
[1., 0., 0.]
]


def inference(input_placeholder):
    with tf.name_scope('inference') as scope:
        W = tf.Variable(tf.zeros([3, 3]), name="weight")
        b = tf.Variable(tf.zeros([3]), name="bias")
        y = tf.nn.softmax(tf.matmul(input_placeholder, W) + b)
    return y

def loss(output, supervisor_labels_placeholder):
    with tf.name_scope('loss') as scope:
        cross_entropy = -tf.reduce_sum(supervisor_labels_placeholder * tf.log(output))
        tf.scalar_summary("x-entropy", cross_entropy)
    return cross_entropy

def training(loss):
    with tf.name_scope('training') as acope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return train_step


#   Variables are defined here to write variables in TensorBoard
with tf.Graph().as_default():
    supervisor_labels_placeholder = tf.placeholder("float", [None,3], name="supervisor_labels_placeholder")
    input_placeholder = tf.placeholder("float", [None, 3], name="input_placeholder")

    #   feed_dict denotes input of placeholder
    feed_dict = {input_placeholder: input, supervisor_labels_placeholder: winning_hands}

    output = inference(input_placeholder)
    loss = loss(output, supervisor_labels_placeholder)
    training_op = training(loss)

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        #   Write placeholders in TensorBoard
        summary_writer = tf.train.SummaryWriter('tensorflow_data', graph_def=sess.graph_def)
        sess.run(init)

        for step in range(1000):
            sess.run(training_op, feed_dict=feed_dict)
            if step % 100 == 0:
                print sess.run(loss, feed_dict=feed_dict)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

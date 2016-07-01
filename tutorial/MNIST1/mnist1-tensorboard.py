# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name="x")

W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

#   クロスエントロピーの計算をname_scopeでまとめる
with tf.name_scope("cross-entropy") as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#   最急降下法の計算をname_scopeでまとめる
with tf.name_scope("training") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#   すべての変数を初期化する準備
init = tf.initialize_all_variables()

#   Sessionを初期化して開始
#   Sessionが、上で定義したNNのグラフ（placeholderとvariableの接続関係）と計算資源（CPU・GPU）をつないでくれる
sess = tf.Session()
sess.run(init)

#   グラフのデータをTensorBoardに書き込む
summary_writer = tf.train.SummaryWriter('mnist1_data', graph_def=sess.graph_def)

#   1000回学習させる
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    #   データを所得
    #   feed_dictを用いて、placeholderに渡す入力データを指定する
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#   学習させた後のNNが、未知のデータを正しく分類できるかの評価基準を設定
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#   correct_predictionは、[True, False, True, True,　・・・]といったデータ構造なので、
#   これを0(False),1(True)に対応付けて、その平均値を計算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

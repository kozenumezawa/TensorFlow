# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#   placeholderは、値（テンソル）が入力される場所
#   [None, 784]は2次元のテンソル。Noneは次元の指定なしの意味
x = tf.placeholder(tf.float32, [None, 784])

#   Variableは変更可能なテンソル
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#   xW+bを計算。yはNN(ニューラルネットワーク)の出力として使う
#   ソフトマックス関数を用いることで、xW+bの各要素を0~1に変換している
y = tf.nn.softmax(tf.matmul(x, W) + b)

#   y_は、教師データやテストデータの正解が入力されるplaceholder
y_ = tf.placeholder(tf.float32, [None, 10])

#   クロスエントロピーはコスト関数
#   y(教師データの入力からNNを用いて求めた出力)とy_（教師データの正解）のクロスエントロピーを計算して、正解度をチェック
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#   最急降下法を用いてクロスエントロピーの最小化を図る
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#   すべての変数を初期化する準備
init = tf.global_variables_initializer()

#   Sessionを初期化して開始
#   Sessionが、上で定義したNNのグラフ（placeholderとvariableの接続関係）と計算資源（CPU・GPU）をつないでくれる
sess = tf.Session()
sess.run(init)

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

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

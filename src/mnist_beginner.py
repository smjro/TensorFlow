# -*- coding: utf-8 -*-

import input_data
import tensorflow as tf

# MNISTのデータセットのダウンロードと読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# シンボリック変数
x = tf.placeholder("float", [None, 784])

# 重みとバイアスを表す変数を用意
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# softmaxの計算
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 訓練時に真のラベルの値を入れるための変数
y_ = tf.placeholder("float", [None,10])

# 交差エントロピーの計算
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 学習方法の定義（ステップサイズ0.01の勾配法で交差エントロピーを最小化）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# セッションの準備
sess = tf.Session()

# 変数の初期化
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
  # 使用するデータの選択
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # 勾配を用いた更新
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 正答率の計算  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 結果の出力
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

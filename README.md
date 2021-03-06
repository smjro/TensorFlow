# TensorFlow
Googleが人工知能・機械学習ソフトウェアTensorFlowをオープンソース化したので、触ってみたいと思います。

## インストール環境
ubuntu 14.04LTS 64bit

## downloadとinstall
OSによって方法が変わります。詳細は[ここ](http://tensorflow.org/get_started/os_setup.md)を参照して下さい。
### Ubuntu/Linux
まず，`pip`をインストールしてください．

> **pip**はPythonで書かれたパッケージソフトウェアをインストール・管理するためのパッケージ管理システムである．

```
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev
```
次に`TensorFlow`をインストールします．
```
# For CPU-only version
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# For GPU-enabled version (only install this version if you have the CUDA sdk installed)
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
```

## VirtualEnv
VirtualEnvを使用したインストールを推奨していたため、再度インストールしました．

※VirtualEnv:Pythonの仮想環境を提供するもの．
```
$ sudo apt-get install python-pip python-dev python-virtualenv
```

次に，virtualenvの環境設定を行います．
```
$ virtualenv --system-site-packages ~/[ディレクトリ名]
$ cd ~/[ディレクトリ名]
```

virtualenvをアクティブにするには以下のコマンドを用います。
```
$ source ~/[ディレクトリ名]/bin/activate  # If using bash
$ source ~/[ディレクトリ名]/bin/activate.csh  # If using csh
(tensorflow)$  # Your prompt should change
```

virtualenv内でTensorFlowをインストールします。
```
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

もし，ここからcloneしてきたのであればインストール中に
`誤ったインタプリタです`と怒られると思います．  
その時は`tensorflow/bin`にある`pip`を次のように書き換えてください．
```
#befor
#!/home/shimadakento/tensorflow/bin/python

#after
#!/home/[ユーザー名]/[ディレクトリ名]/bin/python
```

## TensorFlowを動作させる
python terminalを開きます。
```
$ python

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print sess.run(a+b)
42
>>>
```
上記のように動けばインストール成功です。

## MNIST For ML Beginners
TensorFlowの使い方を覚えるために、チュートリアルをやっていきます。

以下，ソースコードです．詳細は後述します．
```python
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
```

### The MNIST Data
まず手書き数字認識のチュートリアルを始める前に，
使用するデータセットをインポートする必要があります．  
そこで，[ここ](https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py)のコードを`input_data.py`で同じディレクトリに保存してください．  
これで，以下のコードでMNISTデータをインポートできるようになります．
```
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

すべてのMNISTデータは２つの部分（画像とラベル）からなっており，ここでは
画像を`xs`，ラベルを`ys`と呼びます．例えば，訓練画像は`mnist.train.images`
，訓練ラベルは`mnist.train.labels`となっています．

このチュートリアルでは簡単のため画像データを28*28*1=784の数値，
つまり二次元の数字の集まりを横一列に並べたものとしている．

<img src="https://www.tensorflow.org/versions/master/images/MNIST-Matrix.png" width="400px">

`mnist.train.images`は[55000, 784]の配列，`mnist.train.labels`は[55000, 10]の配列です．  
例えば，３というラベルは[0,0,0,1,0,0,0,0,0,0]と表される．

<img src="https://www.tensorflow.org/versions/r0.7/images/mnist-train-ys.png" width="400px">

### Softmax Regressions(ソフトマックス回帰)
softmax回帰はいくつかの異なるものの一つであるオブジェクトに確率を割り当てたいときに用いられます．

ソフトマックス回帰は，次の２ステップで行われます．  
1. 特定のクラスに含まれる入力のevidenceを合計する．  
2. evidenceを確率に変換する．

与えられた画像が特定のクラスに含まれるというevidenceを総計するためにピクセル強度
の加重和を計算します．高い強度を持つピクセルがそのクラス内の画像に反するevidence
であるならば，重みは負となります．一方で，合致すようなevidenceであれば正となります．

具体的には重みは[784, 10]のテンソルであり，画像中の１pixelが０~９である可能性を
表しています．例えば，先の**1**の画像の場合，左上のpixelは[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
となることが多く，これは左上のpixelが０~９のいずれにも影響を及ぼさないことから明らかです．

次の画像はあるモデルがこれらのクラス各々から学んだ重みを示しています．ここで，
赤は負，青は正の重みを表します．

<img src="https://www.tensorflow.org/versions/r0.7/images/softmax-weights.png" width="400px">

また，バイアスと呼ばれるevidenceも加えます．その結果evidenceは次ように表されます．

<img src="https://github.com/smjro/TensorFlow/blob/master/fig/evidence.png" width="200px">

ここで，wは重み，biはクラスiのバイアス，jは入力画像x内のピクセルを加算するための
インデックスです．そして，計算されたevidenceを予測される確率に`softmax関数`によって変換します．

<img src="https://github.com/smjro/TensorFlow/blob/master/fig/probability.png" width="200px">

ここでソフトマックスは線形関数の出力を望みの形に整形する
「活性化」あるいは「リンク」関数として振る舞います
(今回のケースでは１０個の数字の確率分布)．
evidenceの合計を入力が各クラスに含まれる確率に変換するものと考えることができ，
次の式で定義されます．

<img src="https://github.com/smjro/TensorFlow/blob/master/fig/softmax_normalize.png" width="200px">

また，式を拡張すると次式が得られます．

<img src="https://github.com/smjro/TensorFlow/blob/master/fig/softmax.png" width="200px">

１０個の出力の総和が１となることで確立と解釈することが可能になります．  
exp()関数は，値がマイナスにならないように使用され，これにより値がマイナスにならず，
かつ総和が１となるため確率と解釈ができます．

ソフトマックス回帰は次のように表現できます．

<img src="https://www.tensorflow.org/versions/master/images/softmax-regression-scalargraph.png" width="400px">

方程式として書くならば次のように表現できます．

<img src="https://www.tensorflow.org/versions/master/images/softmax-regression-scalarequation.png" width="400px">

行列とベクトルを用いると

<img src="https://www.tensorflow.org/versions/master/images/softmax-regression-vectorequation.png" width="400px">

となり，まとめると

<img src="https://github.com/smjro/TensorFlow/blob/master/fig/softmax_vector.png" width="200px">

と表現できます．

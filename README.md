# TensorFlow
Googleが人工知能・機械学習ソフトウェアTensorFlowをオープンソース化したので、触ってみたいと思います。

## インストール環境
ubuntu 14.04LTS 64bit

## downloadとinstall
OSによって方法が変わります。詳細は[ここ](http://tensorflow.org/get_started/os_setup.md)を参照して下さい。
### Ubuntu/Linux
まず，'pip'をインストールしてください．
> **pip**はPythonで書かれたパッケージソフトウェアをインストール・管理するためのパッケージ管理システムである．
```
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev
```
```
# For CPU-only version
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version (only install this version if you have the CUDA sdk installed)
$ pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

## VirtualEnv
本サイトでVirtualEnvを使用したインストールを推奨していたため、再度インストールしました。

※VirtualEnv:Pythonの仮想環境を提供するもの。
```
$ sudo apt-get install python-pip python-dev python-virtualenv
```

次に、virtualenvの環境設定を行います。ここまでで，`~/tensorflow`ファルダができていると思うので、そこで次のコマンドを入力します。
```
$ virtualenv --system-site-packages ~/tensorflow
$ cd ~/tensorflow
```

virtualenvをアクティブにするには以下のコマンドを用います。
```
$ source bin/activate  # If using bash
$ source bin/activate.csh  # If using csh
(tensorflow)$  # Your prompt should change
```

virtualenv内でTensorFlowをインストールします。
```
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
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

### The MNIST Data
まず手書き数字認識のチュートリアルを始める前に，
使用するデータセットをインポートする必要があります．  
そこで，[ここ](https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py)のコードを'input_data.py'で同じディレクトリに保存してください．  
これで，以下のコードでMNISTデータをインポートできるようになります．
```
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
ダウンロードされたデータは'mnist.train'，'mnist.test'，'mnist.validation'
の３つに分類することができます．  
これは，訓練データのみではなく学習していないデータも使用することにより
一般論を述べれるようにするためです．

すべてのMNISTデータは２つの部分（画像とラベル）からなっており，ここでは
画像を'xs'，ラベルを'ys'と呼びます．例えば，訓練画像は'mnist.train.images'
，訓練ラベルは'mnist.train.labels'となっています．

各画像は28×28ピクセルで，大きな数字の配列として解釈できます．

<img src="https://www.tensorflow.org/versions/master/images/MNIST-Matrix.png" width="400px">

結果としては'mnist.train.images'は[55000, 784]の形をしたテンソル（ｎ次元配列）です．
一次元は画像のインデックス，二次元は各画像のピクセルのインデックスです．
テンソルの各要素は０と１の間のピクセル強度として表されます．

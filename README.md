# TensorFlow
Googleが人工知能・機械学習ソフトウェアTensorFlowをオープンソース化したので、触ってみたいと思います。

## インストール環境
ubuntu 14.04LTS 64bit

## downloadとinstall
OSによって方法が変わります。詳細は[ここ](http://tensorflow.org/get_started/os_setup.md)を参照して下さい。
### Ubuntu/Linux
```
# For CPU-only version
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version (only install this version if you have the CUDA sdk installed)
$ pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

### Mac
```
# Only CPU-version is available at the moment.
$ pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
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

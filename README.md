# Deep Learning from scratch

Book reading log "Deep Learning from scratch" (Ja『ゼロから作る Deep Learning』)

## Summary

### Chapter 1: Introduction to Python

* Basic Python3 syntax
* How to use following libraries
    * numpy
    * matplotlib


### Chapter 2: Perceptron

* Perceptron is algorithm that has input and outputs. when input passes then return specific value
* Perceptron has parameters weight and bias
* Using perceptron, we can implement logical circuit(such as AND/OR gate)
* XOR gate can implement more than two layer perceptron
* single layer perceptron can express only liner area, but multi-layer perceptron can express non-liner area.
* multi-layer perceptron can express computer(logically)


### Chapter 3: Neural Network

* Use sigmoid, ReLU and the other smooth function as activation function
* Using Numpy's multi-dimensional array feature, we can implement neural network efficiently
* Machine learning approach broadly divided classification and regression
* Activation function of output layer
    * Regression: identification function
    * Classification: softmax function
* In classification problem we set output layer numbers equal to classification class number
* Batch: group of input. when we use 


#### notice

##### PIL library installation

`PIL` install via `pip install pillow`

ref: [Python 3.5 対応画像処理ライブラリ Pillow (PIL) の使い方](https://librabuch.jp/blog/2013/05/python_pillow_pil/)

##### MNIST data generation

execute following command

```
$ python download_dataset.py
```

then this script generates `dataset/mnist.pkl`


## Author

[Kota SAITO](https://github.com/noissefnoc)
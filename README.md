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
* Batch: unit of input


#### notice

##### PIL library installation

`PIL` install via `pip install pillow`


##### MNIST data generation

execute following command

```
$ python download_dataset.py
```

then this script generates `dataset/mnist.pkl`


### Chapter 4: Training of Neural Network

* Data sets used in machine learning are divided into training data and test data
* Learning with training data and evaluating the general-purpose ability of the learned model with test data
* Learning of the neural network updates the weight parameter so that the value of the loss function becomes small with the loss function as an index
* When updating the weight parameter, the work of updating the value of the weight in the gradient direction is repeated using the gradient of the weight parameter
* Calculating the derivative by the difference when giving a small value is called numerical differentiation
* The gradient of the weight parameter can be obtained by numerical differentiation
* Calculation by numerical differentiation takes time, but its implementation is simple. On the other hand, the slightly complicated error back propagation method implemented in the next chapter can obtain the gradient at high speed


### notice

I have 1st print of this book. This has many mistakes.

So I check [errata](https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata) and [current version sample code](https://github.com/oreilly-japan/deep-learning-from-scratch).

Some codes doesn't work well now.


### Chapter 5: Backpropagation

* By using the computational graph, it's possible to visually grasp the computation process
* Nodes of computational graph are configured by local calculation
* Propagation in the computational graph performs normal computation. On the other hand, differentiation of each node can be obtained by backpropagation of computational graph.
* By implementing the components of the neural network as layers, it's possible to efficiently compute the gradient calculation(Backpropagation)
* By comparing the results of numerical differentiation and backpropagation method, it can be confirmed that there is no error in implementation of error back propagation method(gradient check)


### Chapter 6: Techies about learning

* Besides SGD, as a method of updating parameters, there are methods such as Momentum, AdaGrad, Adam, etc, which are famous
* The way of giving the initial value of the weight is important for correct learning
* The initial value of Xavier and the initial value of He are valid as the initial value of the weight
* By using Batch Normalization, learning can be advanced quickly, it becomes robust against the initial value
* There are Weight decay and Dropout as normalization methods to suppress over learning
* Searching for hyperparameters is an efficient way to progress while gradually narrowing down the range where good values ​​exist


## References

* [Python 3.5 対応画像処理ライブラリ Pillow (PIL) の使い方](https://librabuch.jp/blog/2013/05/python_pillow_pil/): installation of PIL
* [CS231n](http://cs231n.stanford.edu/syllabus.html)


## Author

[Kota SAITO](https://github.com/noissefnoc)
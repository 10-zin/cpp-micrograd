# C-micrograd

`C-micrograd` is a minimalistic neural network library implemented purely in C. Designed with simplicity in mind, it provides the foundational building blocks to create, train, and evaluate simple multilayer perceptrons (MLPs). 

The repo's concise nature, paired with its neat documentation, makes it ideal for educational purposes, quick prototyping, and for those curious about the inner workings of neural networks and automatic differentiation.

Features
* **Conciseness**: With just around 500 lines of code (LOC), C-micrograd manages to capture the essence of neural network training without unnecessary complexities.
* **Neat Documentation**: Every function and module comes with clear and informative documentation, making it easy to understand and modify.
* **Lightweight and Pure C**: The entire library is written in C without any external dependencies, making it portable and easy to integrate.
* **Automatic Differentiation**: Uses a computation graph-based approach for automatic gradient computation, essential for training neural networks.
* **Modular Design**: Neural networks can be easily constructed using building blocks like Neuron, Layer, and MLP.
* **Gradient Clipping**: An essential feature for preventing exploding gradients, ensuring stable and efficient training.

## Getting started

`engine.h`
This header file serves as the backbone of the C-micrograd library. It encapsulates a suite of primary operations like add, sub, div, mul, each equipped with both forward and backward implementations. Central to this module are the topo_sort and backward functions, which together constitute the crux of the backpropagation mechanism, empowering the library's automatic differentiation capabilities.

`mlp.h`
The mlp.h header is where the fundamental elements of a multilayer perceptron (MLP) reside. It provides a hierarchical structure, starting from individual neurons, building up to neural layers, and culminating in the full-fledged MLP. Each level in this hierarchy offers a deeper abstraction, allowing for the seamless assembly of complex neural architectures.

`train.c` This source file orchestrates the overall training process. By compiling and executing train.c, users can breathe life into the neural network, setting it on a path of learning and adaptation. To train the model:
```
>> gcc -o run_mlp train.c    
>> ./run_mlp
```

We take a simple case-study of predicting if a number is odd or even. `data.txt` stores a few numbers and its labels. 0 for odd and 1 for even. We train the model on this data to check if the model can learn to predict this basic thing. 

While this task may appear elementary, it beautifully exemplifies the neural network's ability to recognize and generalize patterns from data. By training on this dataset, we seek to validate the model's foundational learning capabilities in an intuitive and transparent context.

## Expected Output
Upon executing the training script, you should observe the model's loss decreasing over epochs. This trend signifies the model's learning trajectory, effectively adjusting its weights and biases based on the data. 

A consistent reduction in loss also validates the accuracy of the forward and backward pass implementations in the MLP.

```
EPOCH 0 LOSSS: 1565.061890

EPOCH 1 LOSSS: 877.235229

EPOCH 2 LOSSS: 344.260437

EPOCH 3 LOSSS: 313.122833

EPOCH 4 LOSSS: 379.384338

EPOCH 5 LOSSS: 126.781189

EPOCH 6 LOSSS: 62.352016

EPOCH 7 LOSSS: 15.074145

EPOCH 8 LOSSS: 4.781248

EPOCH 9 LOSSS: 4.902966

EPOCH 10 LOSSS: 7.369284

EPOCH 11 LOSSS: 5.701929

EPOCH 12 LOSSS: 2.986293

EPOCH 13 LOSSS: 3.104216

EPOCH 14 LOSSS: 1.807249

EPOCH 15 LOSSS: 1.715278

EPOCH 16 LOSSS: 2.672203

EPOCH 17 LOSSS: 2.111140
```

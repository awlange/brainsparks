# Brain Sparks

Experimental neural network and deep learning code

## Calrissian

The main neural network code. Makes heavy use of NumPy.

### Implemented and working
- Multi-layer perceptrons (dense layers)
- Backpropagation gradients
- Common activation functions: sigmoid, tanh, ReLU, linear
- Common cost/error functions: MSE, MAE, cross-entropy
- Stochastic gradient descent optimizer

### In the works
- 1-D convolution layer, optional max pooling (gradient nearly done)
- Autoencoders (MLP style)
- GPU acceleration via PyCUDA and/or PyOpenCL (experimenting still)

### Future work
- More optimizer options: rprop, momentum, adagrad, etc.
- 2-D convolution, to follow up on 1-D convolution
- Recurrent networks

## BrainSparks

Experimental data parallelism hook for Calrissian.

- Attempted some linear algebra with Spark, but not much luck.
- Looking into replicated Calrissian networks for parallelism

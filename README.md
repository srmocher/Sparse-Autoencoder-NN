# Sparse Autoencoder Neural network for MNIST handwritten digit recognition

## Description
This project is based on a 3 layer neural network described described as part of the UFLDL tutorial [here](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial).

## Implementation
Things implemented here:
1. Backpropagation algorithm - with and without sparsity penalty. Implementation includes weight decay term
2. Feedforward pass executed before backpropagation phase starts
3. Derivative verification by computing numerical derivative.
4. All implemented using ``numpy`` arrays and output plotted using ``matplotlib``.

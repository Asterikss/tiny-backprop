import numpy as np


class Layer:
  def __init__(self, input_dim, output_dim):
    self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(
      2.0 / input_dim
    )  # He initialization
    self.bias = np.zeros(output_dim)
    self.input = None
    self.output = None

  def forward(self, x):
    self.input = x
    z = x @ self.weights + self.bias
    self.output = np.maximum(0, z)  # ReLU activation
    # self.output = 1. / (1 + np.exp(-z)) # Sigmoid activation
    return self.output

  def backward(self, output_gradient, learning_rate):
    relu_grad = self.output > 0
    dZ = output_gradient * relu_grad
    # sigmoid_grad = self.output * (1. - self.output) # paired with Sigmoid activation
    # dZ = output_gradient * sigmoid_grad

    dW = self.input.T @ dZ
    dB = np.sum(dZ, axis=0)

    self.weights -= learning_rate * dW
    self.bias -= learning_rate * dB

    # Pass the gradient of this layer inputs back to the previous layer
    return dZ @ self.weights.T

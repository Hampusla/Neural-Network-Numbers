import numpy as np
import math, random


class Perceptron:

    def __init__(self, input_size):
        self.weights = np.random.random(input_size)
        self.bias = random.random()

    def activation(self, inputs):
        # Dot product
        n = inputs @ self.weights

        # Sigmoid
        out = self.__sigmoid(n + self.bias)

        return out

    def tune(self, tunes):
        self.bias += tunes[0]
        self.weights = np.add(self.weights, tunes[1])

    def __sigmoid(self, x):
        # return (2 / (1 + math.exp(-x))) - 1
        # return np.sign(x)
        return math.tanh(x)

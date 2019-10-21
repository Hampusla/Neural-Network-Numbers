import math
import random

import numpy as np


class Perceptron:
    """
    An class which simulate a neurons behavior

    ...

    Attributes:
        weights: values which are connected to inputs
        bias: value which are not connected to a input
    """

    def __init__(self, input_size):
        """ Initiates the perceptrons values

        Args:
           input_size: Size for inputs numpy array in activation method
        """

        self.weights = np.random.random(input_size)
        self.bias = random.random()

    def activation(self, inputs):
        """ Takes a numpy array and output a value between -1 and 1

        Do dot product multiplication between inputs and weights.
        Get a output value by appying a tanh function on the dot product and bias added together.

        Args:
            inputs: numpy array of values

        Returns:
            A value between -1 and 1
        """

        n = inputs @ self.weights
        out = self.__function(n + self.bias)
        return out

    def tune(self, tunes):
        """ Update weights and bias

        Adding tunings for each weight and the bias. These tunings are calculated from earlier activation calls.

        Args:
              tunes: List of a value and a numpy array. These are updates for bias and weights.
        """

        self.bias += tunes[0]
        self.weights = np.add(self.weights, tunes[1])

    def __function(self, x):
        return math.tanh(x)

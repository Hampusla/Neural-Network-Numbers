import numpy as np
import math

class Network:

    def __init__(self, input_size):
        self.weights = np.random(1, input_size)


    def activation(self, inputs):

        #Dot product
        n = inputs @ self.weights

        #Sigmoid
        out = self.__sigmoid(n)

        return out

    def tune(self, tunes):


    def __sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


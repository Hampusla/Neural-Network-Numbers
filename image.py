import numpy as np


class Image:

    def __init__(self, image):
        self.array = np.array(map(0.001.__mul__, map(int, image.split(' '))))
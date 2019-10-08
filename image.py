import numpy as np

class Image:


    def __init__(self, image):
        self.array = np.array(map(int, image.split(' ')))
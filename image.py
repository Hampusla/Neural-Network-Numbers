import numpy as np


class Image:

    def __init__(self, image):
        self.array = np.array(list(map((1.0/255).__mul__, map(int, image.split(' ')))))
        self.label = None

    def add_label(self, label):
        self.label = label
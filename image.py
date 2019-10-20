import numpy as np


class Image:
    """
        An class which contains images information

        ...

        Attributes:
            array: pixel values scaled between 0-1
            label: Number which image represent
        """

    def __init__(self, image):
        """ Initiates the image

        images pixels are multiplied so they can be a value from 0-1 instead of 0-255

       Args:
          image: array of pixel values
       """

        self.array = np.array(list(map((1.0/255).__mul__, map(int, image.split(' ')))))
        self.label = None

    def add_label(self, label):
        self.label = label
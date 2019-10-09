import sys
import numpy as np
from image import Image


def validate_arguments():
    if len(sys.argv) < 4:
        print('To few arguments, Missing a dataset')
        return True

    elif len(sys.argv) > 4:
        print('To many arguments')
        return True

    return False


def file_formating(file):
    f = open(file, 'r')
    f = file_scrap(f)

    return f

def image_objecify(file):
    images = []
    for x in range(1000):
        images.append(Image(file.readline()))

    return images

def file_scrap(file):
    for x in range(3):
        file.readline()

    return file


if __name__ == '__main__':

    if validate_arguments():
        sys.exit()

    # Extract images
    f = file_formating(sys.argv[1])

    # Make image objects for all data
    images = image_objecify(f)


    print('Neural Network')

    # train a network of perceptrons (no hidde layers) with the help of patterns (images) and answers (labels)
    # – classify a test set of patterns and return the classifications
    # – The input patterns are stored in a training image file
    # – The correct classifications for these are stored in a training label file
    # – The patterns you should classify are stored in a validation image file – consists of new images
    # – The answers we use to check your result is called the validation label file

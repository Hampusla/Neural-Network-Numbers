import sys
import numpy as np
from image import Image
from perceptron import Perceptron


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

    return np.array(images)

def file_scrap(file):
    for x in range(3):
        file.readline()

    return file

def label_adding(images, file):

    for x in images:
        x.add_label(file.readline())


def split_images(images, size):
    indice = int(1000*size)

    np.random.shuffle(images)
    return np.split(images, [indice, 1000])


def init_network(input_size):
    return {
        '4': Perceptron(input_size),
        '7': Perceptron(input_size),
        '8': Perceptron(input_size),
        '9': Perceptron(input_size),
    }

if __name__ == '__main__':

    if validate_arguments():
        sys.exit()

    # Extract images
    f = file_formating(sys.argv[1])

    #Set size of pictures
    pixel_size = 784

    # Make image objects for all data
    images = image_objecify(f)

    # Close file reader
    f.close()

    # Open label file
    l = file_formating(sys.argv[2])

    # Add labels to all images
    label_adding(images, l)

    # Close file reader
    l.close()

    # Split images in two sets
    test_size = 0.25
    sets = split_images(images, test_size)

    # Initiate perceptrons
    network = init_network(pixel_size)

    print('Neural Network')

    # train a network of perceptrons (no hidde layers) with the help of patterns (images) and answers (labels)
    # – classify a test set of patterns and return the classifications
    # – The input patterns are stored in a training image file
    # – The correct classifications for these are stored in a training label file
    # – The patterns you should classify are stored in a validation image file – consists of new images
    # – The answers we use to check your result is called the validation label file

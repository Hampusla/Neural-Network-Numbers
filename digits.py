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
        images.append(Image(file.readline().rstrip()))

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

def training_cycle(images, network, alpha):
    np.random.shuffle(images)
    for i in images.flat:
        for p in network:
            process_train_image(network[p], i, int(p), alpha)
    # vfunc = np.vectorize(func)
    # vfunc(images, network, alpha)


def func(image, network, alpha):
    for p in network:
        process_train_image(network[p], image, int(p), alpha)

def process_train_image(perceptron, image, number, alpha):
    out = perceptron.activation(image.array)
    err = calc_error(number, image.label,out)
    perceptron.tune(create_tunes(image.array, alpha, err))


def calc_error(num, image_num,guess):
    if num == image_num:
        return 1-guess
    else:
        return -1-guess

def create_tunes(inputs, alpha, error):
    tunes = np.empty(inputs.size)

    for i, t in np.ndenumerate(tunes):
        tune = alpha * error * inputs[i]
        np.put(tunes, i, tune)

    return tunes

def test_cycle(images, network):
    #Add all errors together for each perceptron
    abs_err = 0
    #Get average of error
    for i in images.flat:
        for p in network:
            abs_err = abs_err + abs(process_test_image(network[p], i, int(p)))

    return abs_err
def process_test_image(perceptron, image, number):
    out = perceptron.activation(image.array)
    return calc_error(number, image.label, out)

def goal_reached(goal, err):
    return (goal < err)

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
    print('setup done')

    alpha = 0.01
    goal = 10
    err = goal + 1
    #Train
    # - Input
    # - Calc error
    # - next
    while goal_reached(goal, err):
        print('start training')
        training_cycle(sets[1], network, alpha)

        # Test
        # - Input
        # - Save error
        # - next
        print('start testing')
        err = test_cycle(sets[0], network)
        print('cycle done')


    print('Neural Network')

    # train a network of perceptrons (no hidde layers) with the help of patterns (images) and answers (labels)
    # – classify a test set of patterns and return the classifications
    # – The input patterns are stored in a training image file
    # – The correct classifications for these are stored in a training label file
    # – The patterns you should classify are stored in a validation image file – consists of new images
    # – The answers we use to check your result is called the validation label file

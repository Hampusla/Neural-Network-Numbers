import sys
import numpy as np
from image import Image
from perceptron import Perceptron
import operator


def validate_arguments():
    if len(sys.argv) < 4:
        print('To few arguments, Missing a dataset')
        return True

    elif len(sys.argv) > 4:
        print('To many arguments')
        return True

    return False


def file_formatting(file):
    f = open(file, 'r')

    return file_scrap(f)


def image_objecify(file):
    return np.array([Image(file.readline().rstrip()) for i in range(1000)])

def file_scrap(file):
    for x in range(3):
        file.readline()

    return file

def label_adding(images, file):
    [image.add_label(file.readline()) for image in images]


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
    err = calc_error(number, image.label, out)
    perceptron.tune(create_tunes(image.array, alpha, err))


def calc_error(num, image_num,guess):
    if num == image_num:
        return 1-guess
    else:
        return -1-guess


def create_tunes(inputs, alpha, error):
    return np.multiply(np.multiply(inputs, np.full(inputs.size, alpha)), np.full(inputs.size, error))


def test_cycle(images, network):
    abs_err = 0
    for i in images.flat:
        for p in network:
            abs_err = abs_err + abs(process_test_image(network[p], i, int(p)))

    return abs_err


def process_test_image(perceptron, image, number):
    out = perceptron.activation(image.array)
    return calc_error(number, image.label, out)


def goal_reached(goal, err):
    return (goal < err)


def file_creation(name, out):
    f = open(name, 'w')

    for s in out:
        f.write(str(s) + '\n')

    return f


def validate_numbers(network, images):
    out = []

    for i in images:
        numb = {
            4: 0,
            7: 0,
            8: 0,
            9: 0
        }

        for p in network:
            numb[int(p)] = network[p].activation(i.array)

        out.append((max(numb.items(), key=operator.itemgetter(1))[0]))
    return out

if __name__ == '__main__':

    if validate_arguments():
        sys.exit()

    #Set Parameters for training
    pixel_size = 784
    test_size = 0.20
    alpha = 0.001
    goal = 10

    # Extract training images
    f = file_formatting(sys.argv[1])

    # Make image objects for all data
    images = image_objecify(f)

    # Close file reader
    f.close()

    # Open label file
    l = file_formatting(sys.argv[2])

    # Add labels to all images
    label_adding(images, l)

    # Close file reader
    l.close()

    # Split images in two sets
    sets = split_images(images, test_size)

    # Initiate perceptrons
    network = init_network(pixel_size)

    print('setup done')

    # Iterate training until goal is achieved
    err = goal + 1
    iterations = 0
    while goal_reached(goal, err):
        training_cycle(sets[1], network, alpha)
        err = test_cycle(sets[0], network)
        iterations += 1

    print(iterations)

    # Import validating data
    vf = file_formatting(sys.argv[3])

    # Input to image objects
    validate_images = image_objecify(vf)

    # Run a validation cycle
    out = validate_numbers(network, validate_images)

    # Put answers in a file and then print it
    file_creation('results.txt', out).close()
    print(out)
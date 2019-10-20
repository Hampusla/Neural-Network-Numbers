import sys
import numpy as np
import operator
from image import Image
from perceptron_node import Perceptron

"""Trains four perceptrons on number images then classify new images

Using images of hand drawn numbers (4,7,8,9) to train four separate perceptrons,
each classifying different numbers. After reaching an absolute error of 145 it is done.
After it tries to classify new images and then print the guesses.

"""


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
    indice = int(1000 * size)

    # np.random.shuffle(images)
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


def process_train_image(perceptron, image, number, alpha):
    out = perceptron.activation(image.array)
    err = calc_error(number, int(image.label), out)
    perceptron.tune(create_tunes(image.array, alpha, err))


def calc_error(num, image_num, guess):
    if num == image_num:
        return 1 - guess
    else:
        return -1 - guess


def create_tunes(inputs, alpha, error):
    return [(alpha * error), create_weight_tunes(inputs, alpha, error)]


def create_weight_tunes(inputs, alpha, error):
    return np.multiply(np.multiply(inputs, np.full(inputs.size, alpha)), np.full(inputs.size, error))


def test_cycle(images, network):
    abs_err = 0
    for i in images.flat:
        for p in network:
            abs_err += abs(process_test_image(network[p], i, int(p)))
    return abs_err


def process_test_image(perceptron, image, number):
    out = perceptron.activation(image.array)
    return calc_error(number, int(image.label), out)


def goal_not_reached(goal, err):
    return goal < err


def output_creation(out):
    for s in out:
        print(str(s))


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


# -------- MAIN --------#

if validate_arguments():
    sys.exit()

# Set Parameters for training
pixel_size = 784
test_size = 0.20
alpha = 0.01
goal = 145

# Extract training images
file_images = file_formatting(sys.argv[1])

# Make image objects for all data
images = image_objecify(file_images)

# Close file reader
file_images.close()

# Open label file
file_labels = file_formatting(sys.argv[2])

# Add labels to all images
label_adding(images, file_labels)

# Close file reader
file_labels.close()

# Split images in two sets
sets = split_images(images, test_size)

# Initiate perceptrons
network = init_network(pixel_size)

# Iterate training until goal is achieved
err = goal + 1
while goal_not_reached(goal, err):
    training_cycle(sets[1], network, alpha)
    err = test_cycle(sets[0], network)

# Open file with validation images
file_validation = file_formatting(sys.argv[3])

# Input to image objects
validate_images = image_objecify(file_validation)

# Run a validation cycle
labels = validate_numbers(network, validate_images)

# Print answers
output_creation(labels)

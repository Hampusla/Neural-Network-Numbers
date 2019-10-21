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
    """ Check so the right amount of arguments are sent

   Returns:
       True or False depending on if enough system arguments are sent
   """

    if len(sys.argv) < 4:
        print('To few arguments, Missing a dataset')
        return True

    elif len(sys.argv) > 4:
        print('To many arguments')
        return True

    return False


def file_formatting(file):
    """ Opens a file and remove the extra rows in the start.


    Args:
        file: A image or label file to be opened.

    Returns:
        Open file, ready to read from.
    """

    f = open(file, 'r')
    return file_scrap(f)


def image_objecify(file):
    """ Puts every image pixels in their own image object and that in a numpy array.

    Args:
        file: File with rows of image pixels.
    Returns:
        A numpy array with 1000 image objects.
    """

    return np.array([Image(file.readline().rstrip()) for i in range(1000)])


def file_scrap(file):
    """ Removes the unnecessary lines of a file.

    Args:
        file: A image or label file.
    Returns:
        Same file put only the needed data is left.
    """

    for x in range(3):
        file.readline()

    return file


def label_adding(images, file):
    """ Adds labels to all image objects.

    Args:
        images: Numpy array containing all image objects.
        file: file with labels corresponding to each image.
    """

    [image.add_label(file.readline()) for image in images]


def split_images(images, size):
    """ Splits images into two sets, one for training and one for testing.

    Args:
         images: Numpy array containing all image objects.
         size: Fraction of the numpy array which should be used as test data.
    Returns:
        Two numpy arrays, one which is test data and one is training.
    """

    indice = int(1000 * size)

    np.random.shuffle(images)
    return np.split(images, [indice, 1000])


def init_network(input_size):
    """ Initiates all four perceptrons and put them in a dictionary with their goal number as key.

    Args:
        input_size: Number of pixels in each image.
    Returns:
        Dictionary with four perceptrons, each having their goal number as key.
    """

    return {
        '4': Perceptron(input_size),
        '7': Perceptron(input_size),
        '8': Perceptron(input_size),
        '9': Perceptron(input_size),
    }


def training_cycle(images, network, alpha):
    """ One epoch of training

    Does one epoch of training. This means to for each image give each perceptron its pixels as input and then tune
    its weight depending on the output.

    Args:
        images: Numpy array containing all image objects.
        network: Dictionary with four perceptrons.
        alpha: Learning rate used to slow down how fast the perceptrons learn.
    """

    np.random.shuffle(images)
    for i in images.flat:
        for p in network:
            process_train_image(network[p], i, int(p), alpha)


def process_train_image(perceptron, image, number, alpha):
    """ Trains one perceptron on one image.

    Does training on one perceptron by giving it the images pixels as input. The output is used to calculate the error
    from the what it should have guessed. Lastly tunes the perceptrons weights and bias using the error.

    Args:
         perceptron: Perceptron to train.
         image: Image whose pixels to use as input.
         number: What the perceptron is trained to classify.
        alpha: Learning rate used to slow down how fast the perceptrons learn.
    """

    out = perceptron.activation(image.array)
    err = calc_error(number, int(image.label), out)
    perceptron.tune(create_tunes(image.array, alpha, err))


def calc_error(num, image_num, guess):
    """Calculate error between guees and right answer.

    Args:
        num: What the perceptron is trained to classify.
        image_num: Number in the image.
        guess: Perceptrons guess.
    Returns:
        An error equal to the difference between what the perceptron should have guessed and its guess.
    """

    if num == image_num:
        return 1 - guess
    else:
        return -1 - guess


def create_tunes(inputs, alpha, error):
    """ Creates a list of tuning for a perceptrons bias and weights.

    Args:
        inputs: Numpy array of pixel values used as inputs for a perceptron.
        alpha: Learning rate used to slow down how fast the perceptrons learn.
        error: How wrong the perceptrons output was given inputs as input.
    Returns:
        A list with tuning for the bias and a numpy array containing the weights tuning.
    """

    return [(alpha * error), create_weight_tunes(inputs, alpha, error)]


def create_weight_tunes(inputs, alpha, error):
    """ Creates a array of tuning for a perceptrons weights.

    To make a weight tune the input corresponding to the weight, a learning rate and the error of the perceptron
    is multiplied. To do this in python at a reasonable time matrix multiplication is used. To do this both alpha and
    error is put in arrays as the same size as inputs with all elements the same. These are then matrix multiplied
    to the input array.

    Args:
        inputs: Numpy array of pixel values used as inputs for a perceptron.
        alpha: Learning rate used to slow down how fast the perceptrons learn.
        error: How wrong the perceptrons output was given inputs as input.
    Returns:
        A numpy array of tuning for a perceptrons weights.
    """

    return np.multiply(np.multiply(inputs, np.full(inputs.size, alpha)), np.full(inputs.size, error))


def test_cycle(images, network):
    """ Runs through all test images and add together the error

    Args:
        images: Numpy array containing all image objects.
        network: Dictionary with four perceptrons.
    Returns:
        An absolute error for each image for each perceptron.
    """

    abs_err = 0
    for i in images.flat:
        for p in network:
            abs_err += abs(process_test_image(network[p], i, int(p)))
    return abs_err


def process_test_image(perceptron, image, number):
    """ Test one perceptron on one image

    Args:
        perceptron: Perceptron to train.
        image: Image whose pixels to use as input.
        number: What the perceptron is trained to classify.
    Returns:
        The difference between what the perceptron should have guessed and what it guessed.
    """

    out = perceptron.activation(image.array)
    return calc_error(number, int(image.label), out)


def goal_not_reached(goal, err):
    """ Check if goal error is reached.

    Args:
        goal: Error at which the training should end.
        err: Current error
    Returns:
        True or False depending on if current error is lower than the goal error.
    """
    return goal < err


def validate_numbers(network, images):
    """ Classifies all given images labels using trained perceptrons.

    Classification is done by for each image run all perceptron. The perceptron giving the highest output are said to
    be right and its number is given as classification.

    Args:
         network: Dictionary with four perceptrons.
         images: Numpy array containing all image objects.
    Returns:
        A list of label classifications. from the perceptrons.
    """
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


def output_creation(out):
    """ Prints all classifications to the terminal.

    Args:
        out: List of all classification from validation.
    """
    for s in out:
        print(str(s))



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

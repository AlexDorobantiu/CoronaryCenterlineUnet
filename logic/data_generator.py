# %%
import numpy as np
import copy
import random as random
from logic.simple_itk_utils import save_itk
from logic.image_util import resizeVolume
from logic.dataset_loader import get_data_from_file, get_volume_truth_from_file


def add_data(x_list, y_list, base_data_file, index, imageSizeX, imageSizeY, imageSizeZ):
    data, truth = get_data_from_file(base_data_file, index, imageSizeX, imageSizeY, imageSizeZ, expand_axis = 3)
    x_list.append(data)
    y_list.append(truth)


def convert_data(x_list, y_list):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y


def data_generator(base_data_file, index_list, imageSizeX, imageSizeY, imageSizeZ, batch_size = 1):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)

        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, base_data_file, index, imageSizeX, imageSizeY, imageSizeZ)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list)
                x_list = list()
                y_list = list()


def split_list(input_list, split=0.8, shuffle_list=True, randomNumberGenerator = random.Random(123)):
    if shuffle_list:
        randomNumberGenerator.shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def get_validation_split(data_file_indexes, data_split=0.8, shuffle_list=True, randomNumberGenerator = random.Random(123)):
    """
    Splits the data into the training and validation indices list.
    """
    print("Creating validation split...")
    training_list, validation_list = split_list(
        data_file_indexes, split=data_split,
        shuffle_list = shuffle_list, randomNumberGenerator = randomNumberGenerator)
    return training_list, validation_list


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

def get_training_and_validation_generators(base_data_file, data_file_indexes, imageSizeX, imageSizeY, imageSizeZ, batch_size, data_split=0.8, shuffle_list=True, randomNumberGenerator = random.Random(123)):
    """
    Creates the training and validation generators that can be used when training the model.
    """
    validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file_indexes, data_split=data_split, 
        shuffle_list = shuffle_list, randomNumberGenerator = randomNumberGenerator)

    training_generator = data_generator(base_data_file, training_list, imageSizeX, imageSizeY, imageSizeZ, batch_size=batch_size)
    validation_generator = data_generator(base_data_file, validation_list, imageSizeX, imageSizeY, imageSizeZ, batch_size=validation_batch_size)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(len(training_list), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(len(validation_list), validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


#%%
def testLoadVolumeTruth(index):
    base_data_file = "dataset"

    volumeArrayTruth, originalShape, volumeOrigin, volumeSpacing = get_volume_truth_from_file(base_data_file, index, 80, 80, 64)
    print(originalShape)
    volumeArrayTruth = volumeArrayTruth.astype(np.uint16) * 65535
    print(np.count_nonzero(volumeArrayTruth))
    volumeArrayTruth = resizeVolume(volumeArrayTruth, originalShape[2], originalShape[1], originalShape[0])
    print(np.shape(volumeArrayTruth))
    print(np.count_nonzero(volumeArrayTruth))

    volumeIndex = f'{index:02}'
    datasetDirectory = base_data_file + volumeIndex + "/"
    filename = datasetDirectory + "imageMaskReshaped" + volumeIndex + ".mhd"
    save_itk(volumeArrayTruth, volumeOrigin, volumeSpacing, filename)

# testLoadVolumeTruth(0)

#%%

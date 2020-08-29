# %%
from os import path
from logic.simple_itk_utils import load_itk, save_itk
from logic.image_util import resizeVolume
import numpy as np

# %%

load_thick_truth = True
if load_thick_truth:
    truth_filename = "imageMaskThick{}.mhd"
else:
    truth_filename = "imageMask{}.mhd"

volume_filename = "image{}.mhd"

volumeCache = dict()
volumeTruthCache = dict()

'''
    Returns the image in the shape [z,y,x] with the possibility to resize (NOT crop) to specified size and/or expand on axis
    Example expand_axis = 3, shape will be [z,y,x,1]
    Example expand_axis = 0, shape will be [1,z,y,x]
'''
def get_volume_from_file(base_data_file, index, imageSizeX = None, imageSizeY = None, imageSizeZ = None, expand_axis = None):
    volumeIndex = f'{index:02}'
    datasetDirectory = base_data_file + volumeIndex + "/"
    filename = datasetDirectory + volume_filename.format(volumeIndex)

    if ((filename, imageSizeX, imageSizeY, imageSizeZ) in volumeCache):
        return volumeCache[(filename, imageSizeX, imageSizeY, imageSizeZ)]

    if not path.exists(filename):
        return None, None, None, None
    volumeArray, volumeOrigin, volumeSpacing = load_itk(filename)
    originalShape = np.shape(volumeArray)

    if imageSizeX is not None and imageSizeY is not None and imageSizeY is not None:
        volumeArray = resizeVolume(volumeArray, imageSizeX, imageSizeY, imageSizeZ)
    # volumeArray = volumeArray / 65535.0 # convert to floats only after resizing
    volumeArray = volumeArray / np.max(volumeArray)

    if expand_axis is not None:
        volumeArray = np.expand_dims(volumeArray, axis=expand_axis)
    # print(volumeArray)
    volumeCache[(filename, imageSizeX, imageSizeY, imageSizeZ)] = (volumeArray, originalShape, volumeOrigin, volumeSpacing)
    return volumeCache[(filename, imageSizeX, imageSizeY, imageSizeZ)]


def get_volume_truth_from_file(base_data_file, index, imageSizeX = None, imageSizeY = None, imageSizeZ = None, expand_axis = None):
    volumeIndex = f'{index:02}'
    datasetDirectory = base_data_file + volumeIndex + "/"

    filename = datasetDirectory + truth_filename.format(volumeIndex)
    if ((filename, imageSizeX, imageSizeY, imageSizeZ) in volumeTruthCache):
        return volumeTruthCache[(filename, imageSizeX, imageSizeY, imageSizeZ)]

    if not path.exists(filename):
        return None, None, None, None
    volumeArrayTruth, volumeOrigin, volumeSpacing = load_itk(filename)
    originalShape = np.shape(volumeArrayTruth)

    if imageSizeX is not None and imageSizeY is not None and imageSizeY is not None:
        volumeArrayTruth = resizeVolume(volumeArrayTruth, imageSizeX, imageSizeY, imageSizeZ)

    volumeArrayTruth = volumeArrayTruth > 0

    if expand_axis is not None:
        volumeArrayTruth = np.expand_dims(volumeArrayTruth, axis=expand_axis)
    # print(volumeArrayTruth)
    volumeTruthCache[(filename, imageSizeX, imageSizeY, imageSizeZ)] = (volumeArrayTruth, originalShape, volumeOrigin, volumeSpacing)
    return volumeTruthCache[(filename, imageSizeX, imageSizeY, imageSizeZ)]


def get_data_from_file(base_data_file, index, imageSizeX = None, imageSizeY = None, imageSizeZ = None, expand_axis = None):
    # print("loading dataset %s" % index)
    volume, a, b, c = get_volume_from_file(base_data_file, index, imageSizeX, imageSizeY, imageSizeZ, expand_axis)
    volume_truth, a, b, c = get_volume_truth_from_file(base_data_file, index, imageSizeX, imageSizeY, imageSizeZ, expand_axis)
    return volume, volume_truth

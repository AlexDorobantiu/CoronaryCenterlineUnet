# %%
import numpy as np
from logic.data_generator import get_volume_from_file, get_volume_truth_from_file
from logic.loss_functions import dice_coeff_standard
from logic.simple_itk_utils import save_itk

imageSizeX = 80 #160 # 64 #80 # 512
imageSizeY = 80 #160 # 64 #80 # 512
imageSizeZ = 64 #128 #64 # 32 # 272 # 388

#%%
def test_metrics(index, threshold = 0.5):
    base_data_file = "dataset"
    volumeMask, originalShape, volumeOrigin, volumeSpacing = get_volume_truth_from_file(base_data_file, index)
    print(np.shape(volumeMask))
    print(np.count_nonzero(volumeMask))

    volumeMaskDownsized, a, b, c = get_volume_truth_from_file(base_data_file, index, imageSizeX, imageSizeY, imageSizeZ)
    print(np.shape(volumeMaskDownsized))
    print(np.count_nonzero(volumeMaskDownsized))
    volumeMaskDownsized[volumeMaskDownsized <= threshold] = 0
    volumeMaskDownsized = (volumeMaskDownsized * 32767).astype(np.uint16)
    volumeMaskDownsized = volumeMaskDownsized / np.max(volumeMaskDownsized)

    from logic.image_util import resizeVolume
    volumeMaskUpsized = resizeVolume(volumeMaskDownsized[:, :, :, 0], originalShape[2], originalShape[1], originalShape[0], 
        interpolationOrder = 1)
    print(np.shape(volumeMaskUpsized))
    print(np.count_nonzero(volumeMaskUpsized))

    volumeIndex = f'{index:02}'
    datasetDirectory = base_data_file + volumeIndex + "/"
    filename = datasetDirectory + "imageMaskUpsized" + volumeIndex + ".mhd"
    save_itk(volumeMaskUpsized, volumeOrigin, volumeSpacing, filename)

    print(dice_coeff_standard(volumeMask[:, :, :, 0], volumeMaskUpsized, threshold))

test_metrics(7)


#%%

#%%
import numpy as np
import random as random

from logic.dataset_loader import get_data_from_file

from batchviewer import view_batch
from batchgenerators.dataloading.data_loader import DataLoaderBase
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading import SingleThreadedAugmenter, MultiThreadedAugmenter
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

# %%
def get_training_transform(patch_size):
    training_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    training_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 7 / 360. * 2 * np.pi, 7 / 360. * 2 * np.pi),
            angle_y=(- 7 / 360. * 2 * np.pi, 7 / 360. * 2 * np.pi),
            angle_z=(- 7 / 360. * 2 * np.pi, 7 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    training_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    training_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma transform
    training_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    
    # we can also invert the image, apply the transform and then invert back
    training_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    training_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring
    training_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    # now we compose these transforms together
    training_transforms = Compose(training_transforms)
    return training_transforms

# %%
class DataLoader3D(DataLoader):
    def __init__(self, indices, batch_size, patch_size, base_data_file = "dataset", seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, num_threads_in_multithreaded = 1):

        super().__init__(indices, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        # data is now stored in self._data.
        self.patch_size = patch_size
        self.indices = indices
        self.base_data_file = base_data_file
                            
    
    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        dataset_indices = self.get_indices()
                            
        # initialize empty array for data and seg
        outputData = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        outputTruth = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        
        # iterate over patientIndices and include them in the batch
        for i, j in enumerate(dataset_indices):
            patient_data, patient_truth = get_data_from_file(self.base_data_file, j, expand_axis = 0)

            # crop expects the data to be (b, c, x, y, z), so we need to expand again
            patient_data = np.expand_dims(patient_data, axis=0)
            patient_truth = np.expand_dims(patient_truth, axis=0)

            # commented out the option that the patch must contain something
            # while True:
                # crop expects the data to be (b, c, x, y, z)
                # cropped_patient_data, cropped_patient_seg = crop(patient_data, patient_truth, self.patch_size, crop_type="random")
            #    segmentation_contains_something = np.any(cropped_patient_seg)
            #    if segmentation_contains_something:
            #        break
            cropped_patient_data, cropped_patient_seg = crop(patient_data, patient_truth, self.patch_size, crop_type="random")

            outputData[i] = cropped_patient_data[0]
            outputTruth[i] = cropped_patient_seg[0]


        return {'data': outputData, 'seg': outputTruth}

# %%

def get_augmented_training_generator(base_data_file, data_file_indexes, imageSizeX, imageSizeY, imageSizeZ, batch_size):
    patch_size = (imageSizeZ, imageSizeY, imageSizeX)
    dataloader = DataLoader3D(data_file_indexes, batch_size, patch_size)

    # batch = next(dataloader)
    # print(np.shape(batch['data']))
    # print(np.shape(batch['seg']))
    # view_batch(batch['data'][0], batch['seg'][0], width=320, height=320)

    training_transforms = get_training_transform(patch_size)

    # create transforms that we can actually use for training
    training_generator = SingleThreadedAugmenter(dataloader, training_transforms)
    #tr_gen = MultiThreadedAugmenter(dataloader, tr_transforms, num_processes = 1, num_cached_per_queue=2, seeds=[1234]) # this does not work

    # output is shape (batch_size, 1, z, y, x)
    # batch = next(training_generator)
    # print(np.shape(batch['data']))
    # print(np.shape(batch['seg']))
    # view_batch(batch['data'][0], batch['seg'][0], width=320, height=320)

    return training_generator
# %%

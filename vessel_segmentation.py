# %%
# import os
import numpy as np

from logic.model import get_3d_unet, get_3d_unet_shallow
from logic.data_generator import get_training_and_validation_generators
from logic.dataset_loader import get_volume_from_file, get_volume_truth_from_file
from logic.augmented_data_generator import get_augmented_training_generator
from logic.loss_functions import dice_coeff_standard, dice_coef, dice_coef_loss, dice_coef_negative, combined_dice_coef_and_overlap_loss
from logic.loss_functions import segmentation_overlap_standard, binary_accuracy_standard
from logic.loss_functions import binary_focal_loss, segmentation_overlap_loss, segmentation_overlap, combined_focal_and_overlap_loss
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adagrad
from tensorflow.keras.initializers import RandomNormal, glorot_normal
from tensorflow.keras.metrics import binary_accuracy as metrics_ba, mae as metrics_mae, kld as metrics_kld

from tensorflow.keras import backend as K
K.clear_session()

# %%

maxDatasetSizeX = 512
maxDatasetSizeY = 512
maxDatasetSizeZ = 388

imageSizeX = 96 #80 #128 #80 #160 # 64 #80 # 512
imageSizeY = 96 #80 #128 #80 #160 # 64 #80 # 512
imageSizeZ = 96 #64 #96 #64 #128 #64 # 32 # 272 # 388 # 
modelReduction = 2

modelDirectory = 'models/'
modelName = "model_epoch_{}.h5"

number_of_epochs = 1501
batch_size = 2
dataset_indices = list(range(0, 7))

use_augmented_input = True # False
data_format = "channels_first" if use_augmented_input else "channels_last"


# optimizer=Adam(lr=0.001),
# optimizer=SGD(lr=0.01, momentum=0.9),
# loss=dice_coef_loss,
# loss="binary_crossentropy",
# loss="mean_squared_error",
# metrics=[dice_coef]
# kernel_initializer = RandomNormal(seed = 123)
# kernel_initializer = glorot_normal(seed = 123)

model = get_3d_unet(imageSizeX, imageSizeY, imageSizeZ, data_format, reduction = modelReduction,
# model = get_3d_unet_shallow(imageSizeX, imageSizeY, imageSizeZ, data_format, reduction = modelReduction,
    #optimizer=Adagrad(),
    # optimizer=SGD(lr=0.001, momentum=0.9),
    optimizer=Adam(lr=0.001),
    # loss="binary_crossentropy",
    # loss=dice_coef_loss,
    # loss=binary_focal_loss(alpha=.2, gamma=5),
    # loss=combined_dice_coef_and_overlap_loss,
    loss=combined_focal_and_overlap_loss,
    # loss=segmentation_overlap_loss,
    # loss="kullback_leibler_divergence",
    metrics=[segmentation_overlap, dice_coef, metrics_ba, metrics_kld, metrics_mae],
    print_summary = True
    )


if not use_augmented_input:
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            base_data_file = "dataset",
            data_file_indexes = dataset_indices,
            imageSizeX = imageSizeX, imageSizeY = imageSizeY, imageSizeZ = imageSizeZ,
            batch_size = batch_size,
            data_split = 1)
else:
    train_generator = get_augmented_training_generator(
            base_data_file = "dataset",
            data_file_indexes = dataset_indices,
            imageSizeX = imageSizeX, imageSizeY = imageSizeY, imageSizeZ = imageSizeZ,
            batch_size = batch_size)

# %%
start_training_from_epoch = None

if start_training_from_epoch is not None:
    model.load_weights(modelDirectory + modelName.format(start_training_from_epoch))
    start_epoch = start_training_from_epoch + 1
else:
    start_epoch = 0

saveModelEvery = 1
if not use_augmented_input:
    print("Training on resized data")
    for epochIndex in range(start_epoch, number_of_epochs):
        batches = 0
        for x_batch, y_batch in train_generator:
            # validation_data = next(validation_generator)
            # history = model.fit(x_batch, y_batch, verbose=0, epochs=1, validation_data=validation_data)
            history = model.fit(x_batch, y_batch, verbose=0, epochs=1)
            batches += 1
            if batches >= n_train_steps:
                print('Epoch {}'.format(epochIndex))
                print(history.history)
                # we need to break the loop by hand because the generator loops indefinitely
                break

        if epochIndex % saveModelEvery == 0:
            print('Epoch {} saving'.format(epochIndex))
            model.save_weights(modelDirectory + modelName.format(epochIndex))
else:
    print("Training on augmented data")
    roughNumberOfBlocksPerDataset = (maxDatasetSizeX // imageSizeX) * (maxDatasetSizeY // imageSizeY) * (maxDatasetSizeY // imageSizeY)
    num_batches_per_epoch = roughNumberOfBlocksPerDataset * len(dataset_indices) // batch_size
    for epochIndex in range(start_epoch, number_of_epochs):
        average_loss = 0
        average_segmentation_overlap = 0
        average_dice_coef = 0
        average_binary_accuracy = 0
        
        for batchIndex in range(num_batches_per_epoch):
            batch = next(train_generator)
            history = model.fit(batch['data'], batch['seg'], verbose=0, epochs=1)

            average_loss = average_loss + history.history['loss'][0]
            average_segmentation_overlap = average_segmentation_overlap + history.history['segmentation_overlap'][0]
            average_dice_coef = average_dice_coef + history.history['dice_coef'][0]
            average_binary_accuracy = average_binary_accuracy + history.history['binary_accuracy'][0]

        average_loss = average_loss / num_batches_per_epoch
        average_segmentation_overlap = average_segmentation_overlap / num_batches_per_epoch
        average_dice_coef = average_dice_coef / num_batches_per_epoch
        average_binary_accuracy = average_binary_accuracy / num_batches_per_epoch

        print('Epoch {}'.format(epochIndex))
        print('Average loss: {}'.format(average_loss))
        print('Average segmentation overlap: {}'.format(average_segmentation_overlap))
        print('Average dice: {}'.format(average_dice_coef))
        print('Average binary accuracy: {}'.format(average_binary_accuracy))
        print('Last batch:')
        print(history.history)

        if epochIndex % saveModelEvery == 0:
            print('Epoch {} saving'.format(epochIndex))
            model.save_weights(modelDirectory + modelName.format(epochIndex))

# %%
loadEpoch = 10
model = get_3d_unet(imageSizeX, imageSizeY, imageSizeZ, data_format, reduction = modelReduction)
# model = get_3d_unet_shallow(imageSizeX, imageSizeY, data_format, imageSizeZ, reduction = modelReduction)
model.load_weights(modelDirectory + modelName.format(loadEpoch))

# %%
def predict(model, index, threshold = 0.5):
    base_data_file = "dataset"

    if not use_augmented_input:
        #[imageSizeZ, imageSizeY, imageSizeX]
        volumeArray, originalShape, volumeOrigin, volumeSpacing = get_volume_from_file(base_data_file, index, imageSizeX, imageSizeY, imageSizeZ)
        volumeMask, a, b, c = get_volume_truth_from_file(base_data_file, index, imageSizeX, imageSizeY, imageSizeZ)
        
        #[imageSizeZ, imageSizeY, imageSizeX, 1]
        volumeArray = np.expand_dims(volumeArray, axis=3)
        # print("Non zeroes in original mask: ", np.count_nonzero(volumeMask))

        # expand for batch size
        #[1, imageSizeZ, imageSizeY, imageSizeX, 1]
        volumeArray = np.expand_dims(volumeArray, axis=0)    
        computedVolumeMask = model.predict(volumeArray, verbose=1, batch_size = 1)
        # print("Non zeroes in computed mask: ", np.count_nonzero(volumeMask))

        print(np.shape(volumeArray))
        print(np.shape(volumeMask))
        print(np.shape(computedVolumeMask))

        # trim the unnecessary dimensions
        computedVolumeMask = computedVolumeMask[0, :, :, :, 0]

        from logic.image_util import resizeVolume
        computedVolumeMask = resizeVolume(computedVolumeMask, originalShape[2], originalShape[1], originalShape[0])
    else:
        #[originalSizeZ, originalSizeY, originalSizeX]
        volumeArray, originalShape, volumeOrigin, volumeSpacing = get_volume_from_file(base_data_file, index)
        volumeMask, a, b, c = get_volume_truth_from_file(base_data_file, index)
        
        computedVolumeMask = np.zeros(originalShape, dtype=np.float32)
        print(originalShape)
        maxX = originalShape[2]
        maxY = originalShape[1]
        maxZ = originalShape[0]
        startZ = 0
        while startZ < maxZ:
            if startZ + imageSizeZ > maxZ:
                startZ = maxZ - imageSizeZ
            startY = 0
            while startY < maxY:
                if startY + imageSizeY > maxY:
                    startY = maxY - imageSizeY
                startX = 0
                while startX < maxX:
                    print("startZ %s" % startZ)
                    print("startY %s" % startY)
                    print("startX %s" % startX)
                    if startX + imageSizeX > maxX:
                        startX = maxX - imageSizeX
                    #[1, imageSizeZ, imageSizeY, imageSizeX]
                    volumeArrayBlock = np.expand_dims(volumeArray[startZ:startZ + imageSizeZ, startY:startY + imageSizeY, startX:startX + imageSizeX], axis=0)
                    # expand for batch size
                    #[1, 1, imageSizeZ, imageSizeY, imageSizeX]
                    volumeArrayBlock = np.expand_dims(volumeArrayBlock, axis=0)  

                    computedVolumeMaskBlock = model.predict(volumeArrayBlock, verbose=1, batch_size = 1)
                    # trim the unnecessary dimensions
                    computedVolumeMaskBlock = computedVolumeMaskBlock[0, 0, :, :, :]

                    computedVolumeMask[startZ:startZ + imageSizeZ, startY:startY + imageSizeY, startX:startX + imageSizeX] = computedVolumeMaskBlock
                    startX = startX + imageSizeX
                startY = startY + imageSizeY
            startZ = startZ + imageSizeZ


    if volumeMask is not None:
        dice_all = []
        for impred, imtest in zip(computedVolumeMask, volumeMask):
            dice_all.append(dice_coeff_standard(imtest, impred, threshold))
        print("Dice: ", np.mean(dice_all))
    print("Overlap ", segmentation_overlap_standard(volumeMask, computedVolumeMask))
    print("Accuracy ", binary_accuracy_standard(volumeMask, computedVolumeMask))

    computedVolumeMask[computedVolumeMask <= threshold] = 0
    computedVolumeMask = (computedVolumeMask * 16383).astype(np.uint16)

    from logic.simple_itk_utils import save_itk

    print(np.shape(computedVolumeMask))
    print(np.count_nonzero(computedVolumeMask))

    volumeIndex = f'{index:02}'
    datasetDirectory = base_data_file + volumeIndex + "/"
    filename = datasetDirectory + "imageMaskPredicted" + volumeIndex + ".mhd"
    save_itk(computedVolumeMask, volumeOrigin, volumeSpacing, filename)


# predict(model, 6)
predict(model, 7, threshold = 0.0)

#%%
def evaluateModelOnValidation(model_directory, last_epoch, start_epoch = 0, increment = 5, dataset_index = 7):
    base_data_file = "dataset"
    model = get_3d_unet(imageSizeX, imageSizeY, imageSizeZ, data_format, reduction = modelReduction)
    epoch_index = start_epoch

    if not use_augmented_input:
        volumeMask, a, b, c = get_volume_truth_from_file(base_data_file, dataset_index, imageSizeX, imageSizeY, imageSizeZ)
    else:
        volumeMask, a, b, c = get_volume_truth_from_file(base_data_file, dataset_index)

    while epoch_index <= last_epoch:
        model.load_weights(model_directory + modelName.format(epoch_index))
    

        if not use_augmented_input:
            volumeArray, originalShape, volumeOrigin, volumeSpacing = get_volume_from_file(base_data_file, dataset_index, imageSizeX, imageSizeY, imageSizeZ)
            volumeArray = np.expand_dims(volumeArray, axis=3)
            volumeArray = np.expand_dims(volumeArray, axis=0)    
            computedVolumeMask = model.predict(volumeArray, verbose=0, batch_size = 1)
            # trim the unnecessary dimensions
            computedVolumeMask = computedVolumeMask[0, :, :, :, 0]

            from logic.image_util import resizeVolume
            computedVolumeMask = resizeVolume(computedVolumeMask, originalShape[2], originalShape[1], originalShape[0])
        else:
            #[originalSizeZ, originalSizeY, originalSizeX]
            volumeArray, originalShape, volumeOrigin, volumeSpacing = get_volume_from_file(base_data_file, dataset_index)
            
            
            computedVolumeMask = np.zeros(originalShape, dtype=np.float32)
            maxX = originalShape[2]
            maxY = originalShape[1]
            maxZ = originalShape[0]
            startZ = 0
            while startZ < maxZ:
                if startZ + imageSizeZ > maxZ:
                    startZ = maxZ - imageSizeZ
                startY = 0
                while startY < maxY:
                    if startY + imageSizeY > maxY:
                        startY = maxY - imageSizeY
                    startX = 0
                    while startX < maxX:
                        if startX + imageSizeX > maxX:
                            startX = maxX - imageSizeX
                        #[1, imageSizeZ, imageSizeY, imageSizeX]
                        volumeArrayBlock = np.expand_dims(volumeArray[startZ:startZ + imageSizeZ, startY:startY + imageSizeY, startX:startX + imageSizeX], axis=0)
                        # expand for batch size
                        #[1, 1, imageSizeZ, imageSizeY, imageSizeX]
                        volumeArrayBlock = np.expand_dims(volumeArrayBlock, axis=0)  

                        computedVolumeMaskBlock = model.predict(volumeArrayBlock, verbose=0, batch_size = 1)
                        # trim the unnecessary dimensions
                        computedVolumeMaskBlock = computedVolumeMaskBlock[0, 0, :, :, :]

                        computedVolumeMask[startZ:startZ + imageSizeZ, startY:startY + imageSizeY, startX:startX + imageSizeX] = computedVolumeMaskBlock
                        startX = startX + imageSizeX
                    startY = startY + imageSizeY
                startZ = startZ + imageSizeZ

        print("Epoch index:", epoch_index)
        print("Dice: ", dice_coeff_standard(volumeMask, computedVolumeMask))
        print("Overlap ", segmentation_overlap_standard(volumeMask, computedVolumeMask))
        print("Accuracy ", binary_accuracy_standard(volumeMask, computedVolumeMask))
        
        epoch_index += increment


evaluateModelOnValidation("models 128x128x96 batch 1 model div 1/", 
    260, start_epoch=0, increment=5, dataset_index=7)

# %%
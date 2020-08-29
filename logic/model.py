# %%
from tensorflow.keras import Model, layers, backend
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPool3D, UpSampling3D
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adagrad
from tensorflow.keras.initializers import RandomNormal, glorot_normal
from logic.loss_functions import dice_coef, dice_coef_loss

# 3D Unet
def get_3d_unet(imageSizeX, imageSizeY, imageSizeZ, data_format="channels_last", reduction = 2,
    optimizer = Adam(lr=0.001), loss = dice_coef_loss, metrics=[dice_coef], 
    kernel_initializer = glorot_normal(seed = 123),
    print_summary = False):

    backend.set_image_data_format(data_format)  # TF dimension ordering in this code

    ## 1 means 1 channel (grayscale)
    if data_format == "channels_last":
        input_shape=(imageSizeZ, imageSizeY, imageSizeX, 1)
        concatenate_axis = 4
    else:
        input_shape=(1, imageSizeZ, imageSizeY, imageSizeX)
        concatenate_axis = 1

    inputs = Input(input_shape)

    ## going down
    conv1 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', 
        input_shape=input_shape, data_format = data_format, kernel_initializer = kernel_initializer)(inputs)
    conv1 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv1')(conv1)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(pool1)
    conv2 = Conv3D(64 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(conv2)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(pool2)
    conv3 = Conv3D(128 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv3')(conv3)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(pool3)
    conv4 = Conv3D(256 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(conv4)
    pool4 = MaxPool3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(pool4)
    conv5 = Conv3D(512 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv5')(conv5)


    # going up
    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=concatenate_axis)
    conv6 = Conv3D(256 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(up6)
    conv6 = Conv3D(256 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2),)(conv6), conv3], axis=concatenate_axis)
    conv7 = Conv3D(128 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(up7)
    conv7 = Conv3D(128 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv7')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=concatenate_axis)
    conv8 = Conv3D(64 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(up8)
    conv8 = Conv3D(64 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=concatenate_axis)
    conv9 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(up9)
    conv9 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv9')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_initializer = kernel_initializer)(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    if print_summary:
        model.summary(line_length = 120)
    return model

# 3D Unet
def get_3d_unet_shallow(imageSizeX, imageSizeY, imageSizeZ, data_format="channels_last", reduction = 2,
    optimizer = Adam(lr=0.001), loss = dice_coef_loss, metrics=[dice_coef], kernel_initializer = glorot_normal(seed = 123),
    print_summary = False):

    backend.set_image_data_format(data_format)  # TF dimension ordering in this code

    ## 1 means 1 channel (grayscale)
    input_shape=(imageSizeZ, imageSizeY, imageSizeX, 1)

    inputs = Input(input_shape)

    ## going down
    conv1 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', 
        input_shape=input_shape, data_format = data_format, kernel_initializer = kernel_initializer)(inputs)
    conv1 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv1')(conv1)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)

  
    conv5 = Conv3D(128 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(pool1)
    conv5 = Conv3D(128 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv5')(conv5)


    # going up
    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv1], axis=4)
    conv9 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer)(up9)
    conv9 = Conv3D(32 // reduction, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_initializer, name='conv9')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_initializer = kernel_initializer)(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    if print_summary:
        model.summary(line_length = 120)
    return model

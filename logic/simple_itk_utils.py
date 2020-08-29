# %%

import SimpleITK as sitk
import numpy as np

'''
This function reads an ITK file ('.mhd' and maybe others) using SimpleITK and return the image array, origin and spacing of the image.
image array has the shape [z, y, x]
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a numpy array first and then shuffle the dimensions to get axis
    # in the order z,y,x
    imageArray = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = itkimage.GetOrigin()

    # Read the spacing along each dimension
    spacing = itkimage.GetSpacing()

    return imageArray, origin, spacing

def save_itk(imageArray, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(imageArray, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)

def convertDoubleVectorToNumpyArray(doubleVector):
    return np.array(list(reversed(doubleVector)))

def convertNumpyArrayToDoubleVector(npArray):
    return tuple(np.flip(npArray, 0))

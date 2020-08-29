# %%
from logic.simple_itk_utils import load_itk, save_itk
import numpy as np

def convertSpatialCoordinatesToArrayCoordinates(x, y, z, volumeOrigin, volumeSpacing):
    # the values are truncated
    outputX = int((x - volumeOrigin[0]) / volumeSpacing[0])
    outputY = int((y - volumeOrigin[1]) / volumeSpacing[1])
    outputZ = int((z - volumeOrigin[2]) / volumeSpacing[2])
    return outputX, outputY, outputZ

maxValue = 65535 # uint16 maxValue
lineThickness = 2 # 0 means single pixel, 1 means cube 3 by 3, 2 means cube 5 by 5

if lineThickness > 0:
    outputFilename = "imageMaskThick{}.mhd"
else:
    outputFilename = "imageMask{}.mhd"

for n in range(0, 8):
    volumeIndex = f'{n:02}'
    datasetDirectory = "dataset" + volumeIndex + "/"
    volumeArray, volumeOrigin, volumeSpacing = load_itk(datasetDirectory + "image" + volumeIndex + ".mhd")
    outputVolumeArray = np.zeros(volumeArray.shape, np.uint16)
    
    # print(volumeArray.shape)
    for vesselIndex in range(0, 4):
        vesselDirectory = datasetDirectory + "vessel" + str(vesselIndex) + "/"
        vesselReferenceFile = vesselDirectory + "reference.txt"
        with open(vesselReferenceFile) as input_file:
            for line in input_file:
                floats = line.strip().split()
                x = float(floats[0])
                y = float(floats[1])
                z = float(floats[2])
                x1, y1, z1 = convertSpatialCoordinatesToArrayCoordinates(x, y, z, volumeOrigin, volumeSpacing)
                # print(x1, y1, z1)
                for deltaZ in range(-lineThickness, lineThickness + 1):
                    for deltaY in range(-lineThickness, lineThickness + 1):
                        for deltaX in range(-lineThickness, lineThickness + 1):
                            outputVolumeArray[z1 + deltaZ, y1 + deltaY, x1 + deltaX] = maxValue
    
    # print(outputVolumeArray)
    save_itk(outputVolumeArray, volumeOrigin, volumeSpacing, datasetDirectory + outputFilename.format(volumeIndex))

#%%

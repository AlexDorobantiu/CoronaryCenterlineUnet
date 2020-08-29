# %%
import numpy as np
import scipy.ndimage

def resizeVolume(a, newSizeX, newSizeY, newSizeZ, interpolationOrder = None):
    shape = np.shape(a)
    zoomX = newSizeX / shape[2]
    zoomY = newSizeY / shape[1]
    zoomZ = newSizeZ / shape[0]
    if interpolationOrder is None:
        if zoomX < 1 or zoomY < 1 or zoomZ < 1:
            interpolationOrder = 1
        else:
            interpolationOrder = 1
    result = scipy.ndimage.zoom(a, zoom=(zoomZ, zoomY, zoomX), order = interpolationOrder)
    return result


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray
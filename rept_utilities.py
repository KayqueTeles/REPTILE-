""" Utility functions. """
import numpy as np, os, random, shutil, sklearn, keras, wget, zipfile, tarfile, matplotlib.pyplot as plt, bisect, cv2, tensorflow as tf, sys

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers import Input
from collections import Counter
from keras import models
from keras.models import Sequential
from keras.layers import Convolution2D
from tensorflow.keras.models import Model
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.regularizers import l2

####################################################3#############
##############SOME NECESSARY FUNCTIONS############################
##################################################################


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def save_image(vector, version, index, step, input_size):
    
    image = vector 
    index = index + 1
    image = toimage(image)
    image.save("save_im_full_ver_%s_ind_%s.png" % (version, index))
    return [image, index]

def save_clue(x_data, y_data, TR, version, step, input_shape, nrows, ncols, index):

    figcount = 0
    plt.figure()
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,20))
    for i in range(nrows):
        for j in range(ncols):
            temp_image = toimage(np.array(x_data[figcount, :, :, :]))
            axs[i, j].imshow(temp_image)
            axs[i, j].set_title('Class: %s' % y_data[figcount])
            figcount = figcount + 1

    index = index + 1
    plt.show()
    plt.savefig("CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png". format(TR, version, step, input_shape, input_shape, index))
    return figcount

def fileremover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling, architecture, learning_rate):

    piccounter = 0
    print('\n ** Cleaning up previous files...')
    if os.path.exists('./Code_data_version_%s.csv' % version):
        os.remove('./Code_data_version_%s.csv' % version)
        piccounter = piccounter + 1
    if os.path.exists('./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./Losses_MSE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./Losses_MSE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./train_Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./train_Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./TrainTest_rate_TR_{}.png'. format(TR)):
        os.remove('./TrainTest_rate_TR_{}.png'. format(TR))
        piccounter = piccounter + 1
    if os.path.exists('./model_REPTILE_version_%s.png' % version):
        os.remove('./model_REPTILE_version_%s.png' % version)
        piccounter = piccounter + 1
    if os.path.exists('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1    
    if os.path.exists('./ROCLensDetectNet_Full_%s.png' % TR):
        os.remove('./ROCLensDetectNet_Full_%s.png' % TR)
        piccounter = piccounter + 1    
    if os.path.exists('./Code_data_version_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}.csv'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture)):
        os.remove('./Code_data_version_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}.csv'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        piccounter = piccounter + 1
    
    for lo in range(10):
        for by in range(TR*10):   
            if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, by)):
                os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, by))
                piccounter = piccounter + 1  
            if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}_stage_{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize, by)):
                os.remove('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}_stage_{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize, by))
                piccounter = piccounter + 1   
            if os.path.exists('./Color_Hist_IMG {}_index_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(lo, by, TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
                os.remove('./Color_Hist_IMG_{}_index_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(lo, by, TR, shots, input_shape, input_shape, meta_iters, version, normalize))
                piccounter = piccounter + 1  
    phase = ("train", "test")
    step = ("Before Norm", "Normalized", "Resized", "Resized & Normalized")
    for i in phase:
        for j in step:
            if os.path.exists("Color_Hist_IMG_{}_step_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(i, j, TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
                os.remove("Color_Hist_IMG_{}_step_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(i, j, TR, shots, input_shape, input_shape, meta_iters, version, normalize))
                piccounter = piccounter + 1

    print(" ** Removing done. %s files removed." % (piccounter))

    if os.path.exists("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture)):
        shutil.rmtree("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
    

def filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling, architecture, learning_rate):

    print('\n ** Moving created files to a certain folder.')
    counter = 0
    print(" ** Checking if there's a GRAPHS folder...")
    if os.path.exists('REPT-GRAPHS'):
        print(" ** GRAPHS file found. Moving forward.")
    else:
        print(" ** None found. Creating one.")
        os.mkdir('REPT-GRAPHS')
        print(" ** Done!")
    print(" ** Checking if there's an REP folder...")
    if os.path.exists("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        os.mkdir("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        print(" ** Done!")

    if os.path.exists("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}/SAMPLES". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}/SAMPLES". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        os.mkdir("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}/SAMPLES". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir("REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}/SAMPLES". format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
        print(" ** Done!")

    dest1 = ('/home/kayque/LENSLOAD/REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))
    dest2 = ('/home/kayque/LENSLOAD/REPT-GRAPHS/REP_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}/SAMPLES'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture))

    if os.path.exists('./Code_data_version_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}.csv'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture)):
        shutil.move('./Code_data_version_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}.csv'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture), dest1)
        counter = counter + 1
    if os.path.exists('./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move("./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move("./Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./Losses_MSE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move("./Losses_MSE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./train_Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move("./train_Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./TrainTest_rate_TR_{}.png'. format(TR)):
        shutil.move('./TrainTest_rate_TR_{}.png'. format(TR), dest1)
        counter = counter + 1
    if os.path.exists('./ROCLensDetectNet_Full_%s.png' % TR):
        shutil.move('./ROCLensDetectNet_Full_%s.png' % TR, dest1)
        counter = counter + 1
    if os.path.exists('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./model_REPTILE_version_%s.png' % version):
        shutil.move('./model_REPTILE_version_%s.png' % version, dest1)
        counter = counter + 1
    
    for lo in range(10):
        for b in range(TR*10):   
            if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, b)):
                shutil.move('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, b), dest2)
                counter = counter + 1   
            if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}_stage_{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize, b)):
                shutil.move('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}_stage_{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize, b), dest1)
                counter = counter + 1
    
    phase = ("train", "test")
    step = ("Before Norm", "Normalized", "Resized", "Resized & Normalized")
    for i in phase:
        for j in step:
            if os.path.exists("Color_Hist_IMG_{}_step_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(i, j, TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
                shutil.move("Color_Hist_IMG_{}_step_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(i, j, TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest2)
                counter = counter + 1
    print(" ** Moving done. %s files moved." % counter)
    print(dest1)

def ROCCurveCalculate(y_test, x_test, model):

    probs = model.predict(x_test)
    #probsp = probs
    probsp = probs[:, 1]
    y_new = y_test  #[:, 1]
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))
    
    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        TPscore, FPscore, TNscore, FNscore = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
            if probsp[xz] > thresh:
                if y_new[xz] == 1:                
                    TPscore = TPscore + 1
                else:
                    FPscore = FPscore + 1
            else:
                if y_new[xz] == 0:
                    TNscore = TNscore + 1
                else:
                    FNscore = FNscore + 1
        TPRate = TPscore / (TPscore + FNscore)
        FPRate = FPscore / (FPscore + TNscore)
        tpr.append(TPRate)
        fpr.append(FPRate)           

    auc2 = roc_auc_score(y_test, probsp)
    auc = metrics.auc(fpr, tpr)
    print('\n ** AUC (via metrics.auc): %s, AUC (via roc_auc_score): %s' % (auc, auc2))
    return [tpr, fpr, auc, auc2, thres]

def data_downloader():
    print('\n ** Checking dataset files...')
    if os.path.exists('./lensdata/x_data20000fits.h5'):
        print(" ** Files from lensdata.tar.gz were already downloaded.")
    else:
        print("n ** Downloading lensdata.zip...")
        wget.download('https://clearskiesrbest.files.wordpress.com/2019/02/lensdata.zip')
        print(" ** Download successful. Extracting...")
        with zipfile.ZipFile("lensdata.zip", 'r') as zip_ref:
            zip_ref.extractall() 
            print(" ** Extracted successfully.")
        print(" ** Extracting data from lensdata.tar.gz...")
        tar = tarfile.open("lensdata.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        print(" ** Extracted successfully.")
    if os.path.exists('./lensdata/x_data20000fits.h5'):
        print(" ** Files from lensdata.tar.gz were already extracted.")
    else:
        print(" ** Extracting data from #DataVisualization.tar.gz...")     
        tar = tarfile.open("./lensdata/DataVisualization.tar.gz", "r:gz")
        tar.extractall("./lensdata/")
        tar.close()
        print(" ** Extracted successfully.")
        print(" ** Extrating data from x_data20000fits.h5.tar.gz...")     
        tar = tarfile.open("./lensdata/x_data20000fits.h5.tar.gz", "r:gz")
        tar.extractall("./lensdata/")
        tar.close()
        print(" ** Extracted successfully.") 
    if os.path.exists('lensdata.tar.gz'):
            os.remove('lensdata.tar.gz')
    if os.path.exists('lensdata.zip'):
            os.remove('lensdata.zip')
    for pa in range(0, 10, 1):
        if os.path.exists('lensdata ({}).zip'. format(pa)):
            os.remove('lensdata ({}).zip'. format(pa))

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        print(sum_score)
        summ = actual[i] * np.log(1e-15 + predicted[i])
        sum_score = sum_score + summ
        print(predicted[i])
        print(actual[i])
        mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

def mean_squared_error(actual, predicted):
    sum_square_error = 0.0
    for i in range(len(actual)):
        sum_square_error += (actual[i] - predicted[i])**2.0
    mean_square_error = 1.0 / len(actual) * sum_square_error
    return mean_square_error

def categorical_cross_entropy_loss(actual, predicted):
	sum_score = 0.0
	for i in range(len(actual)):
		for j in range(len(actual[i])):
			sum_score += actual[i][j] * log(1e-15 + predicted[i][j])
	mean_sum_score = 1.0 / len(actual) * sum_score
	return -mean_sum_score

####################################################3#############
##################################################################F

def FScoreCalc(y_test, x_test, model):

    probsp = np.argmax(model.predict(x_test), axis=-1)
    y_test = np.argmax(y_test, axis =-1)

    f_1_score = sklearn.metrics.f1_score(y_test, probsp)
    f_001_score = sklearn.metrics.fbeta_score(y_test, probsp, beta=0.01)
    
    print('\n ** F1_Score: %s, F0.01_Score: %s.' % (f_1_score, f_001_score))
    return [f_1_score, f_001_score]

def conv_window(vector):
    window_length = 100   #ORIGINALLY 100
    vec_s = np.r_[
        vector[window_length - 1 : 0 : -1], vector, vector[-1:-window_length:-1]
    ]
    w = np.hamming(window_length)
    vec_y = np.convolve(w / w.sum(), vec_s, mode="valid")
    return vec_y

def roc_curve_graph(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, f1_score, f001_score):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--') # k = color black
    plt.plot(fpr, tpr, label="AUC: %.3f, F1: %.3f, F001: %.3f." % (auc, f1_score, f001_score), linewidth=3) # for color 'C'+str(j), for j[0 9]
    plt.legend(loc='lower right', ncol=1, mode="expand")
    plt.title('ROC')
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.savefig("ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))

def roc_curve_graph_series(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, arg):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--') # k = color black
    plt.plot(fpr, tpr, label="AUC: %.3f" % auc, linewidth=3) # for color 'C'+str(j), for j[0 9]
    plt.legend(loc='lower right', ncol=1, mode="expand")
    plt.title('ROC for character %s' % arg)
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.savefig("ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}_stage_{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize, arg))

def acc_graph(test_y, train_y, TR, shots, input_shape, meta_iters, version, normalize):
    plt.figure()
    x = np.arange(0, len(test_y), 1)
    plt.plot(x, test_y, x, train_y)
    plt.legend(["test", "train"])
    plt.title('Accuracies')
    plt.grid()
    plt.savefig("Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))

def loss_graph(tra_loss, tes_loss, TR, shots, input_shape, meta_iters, version, normalize):
    plt.figure()
    x = np.arange(0, int(len(tra_loss)), 1)
    plt.plot(x, tra_loss, x, tes_loss)
    plt.legend(["test", "train"])
    plt.title('Losses')
    plt.grid()
    plt.savefig("Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))

def examples_graph(rows, cols, train_dataset, index, TR, shots, input_shape, meta_iters, version, normalize):
    _, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
    sample_keys = list(train_dataset.data.keys())

    for a in range(rows):
        for b in range(cols):
            temp_image = train_dataset.data[sample_keys[a]][b]
            temp_image = toimage(temp_image)
            if b == 2:
                axarr[a, b].set_title("Class : " + sample_keys[a])
            axarr[a, b].imshow(temp_image)
            axarr[a, b].xaxis.set_visible(False)
            axarr[a, b].yaxis.set_visible(False)
    plt.show()
    plt.savefig("EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))

def miniimagenet_downloader():
    print('\n ** Checking dataset files...')
    if os.path.exists('./mini-imagenet-cache-train.pkl'):
        print(" ** Dataset ready!")
    else:
        if os.path.exists('./miniimagenet.zip'):
            print(" ** Files were already downloaded but not extracted.")
            print(" ** Extracting...")
            with zipfile.ZipFile("miniimagenet.zip", 'r') as zip_ref:
                zip_ref.extractall() 
            print(" ** Extracted successfully.")
            if os.path.exists('miniimagenet.zip'):
                    os.remove('miniimagenet.zip')
        else:
            print("n ** None found. Downloading mini-imagenet files.zip...")
            wget.download('https://data.deepai.org/miniimagenet.zip')
            print(" ** Download successful. Extracting...")
            with zipfile.ZipFile("miniimagenet.zip", 'r') as zip_ref:
                zip_ref.extractall() 
            print(" ** Extracted successfully.")
            if os.path.exists('miniimagenet.zip'):
                    os.remove('miniimagenet.zip')

def get_images(paths, labels, nb_samples=None, shuffle=True):
    print(' -- get_images being used.')
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

def distrib_graph(y_data, y_val, y_test, classes, TR):
    trainval_count = [np.count_nonzero(y_data == 1)+np.count_nonzero(y_val == 1), np.count_nonzero(y_data == 0)+np.count_nonzero(y_val == 0)]
    test_count = [np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0)]
    width = 0.35

    #############DISTRIBUTION GRAPH#########
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(classes, test_count, width, label='Test')
    ax.bar(classes, trainval_count, width, bottom=test_count, label='Train+Val')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset distribution')
    ax.legend(loc='lower right')
    fig.savefig("TrainTest_rate_TR_{}.png". format(TR))

def class_choose(y_data, x):
    y_d = (y_data == x)
    return y_d

def extraction(image, label):
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [56, 56])
    return image, label

def resize_image(x_datas, y_data, input_shape, num_channels):
    print("\n ** Resizing images...")
    x_data = np.zeros(shape=(1,4))
    for y in range(len(y_data)):
        for x in range(num_channels):
            image = Image.fromarray(x_datas[y,:,:,x])
            image.resize(size=(input_shape, input_shape))
            image = np.asarray(image)
            x_data = np.append(x_data, np.array([image]), axis=2)
            x_data = np.array(x_data)
    print(" ** resized to: ", x_data.shape)
    return x_data

def basic_conv_model(normalize, dropout, maxpooling, activation_layer, output_layer, input_shape, num_channels, filters, kernel_size, padding, learning_rate, optimizer, num_classes):
    def conv_bn(x):
        x = layers.Conv2D(filters=64, kernel_size=5, padding="same")(x)
        if normalize == "BatchNormalization":
            x = layers.BatchNormalization()(x)
        if maxpooling == "yes":
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout)(x)
        if activation_layer == "relu":
            x = layers.ReLU()(x)
            return x
        elif activation_layer == "softmax":
            return keras.activations.softmax(x)
        elif activation_layer == "sigmoid":
            return keras.activations.sigmoid(x)

    inputs = layers.Input(shape=(input_shape, input_shape, num_channels))
    x = conv_bn(inputs)
    x = conv_bn(x)
    x = conv_bn(x)
    x = conv_bn(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation=output_layer)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    return model

def multiclass_roc_graphs(num_classes, y_test, x_test, model, TR, shots, input_shape, meta_iters, version, normalize):
    print(" ** Generating roc graphs...")
    lauc, AUCall, FPRall, TPRall, f1s, f001s = ([] for i in range(6))
    for j in range(num_classes):
        y_test = (y_test == j)
        print(test_l)
        tpr, fpr, auc, auc2, thres = ROCCurveCalculate(y_test, x_test, model)
        lauc = np.append(lauc, auc)
        AUCall.append(auc2)
        FPRall.append(fpr)
        TPRall.append(tpr)
        roc_curve_graph_series(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, j)
    print('\n ** Generating ultimate ROC graph...')
    medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--') # k = color black

    mauc = np.percentile(lauc, 50.0)
    mAUCall = np.percentile(AUCall, 50.0)
    plt.title('Median ROC over %s characters' % (num_classes))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)

    for num in range(0,int(thres),1):
        lis = [item[num] for item in TPRall]
        los = [item[num] for item in FPRall]
            
        medians_x.append(np.percentile(los, 50.0))
        medians_y.append(np.percentile(lis, 50.0))
        lowlim.append(np.percentile(lis, 15.87))
        highlim.append(np.percentile(lis, 84.13))
        
    lowauc = metrics.auc(medians_x, lowlim)
    highauc = metrics.auc(medians_x, highlim)

    print(lowauc, mAUCall, highauc)

    plt.plot(medians_x, medians_y, 'b', label = 'AUC: %s' % mauc, linewidth=3)  
    plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha=0.3, interpolate=True)
    plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha=0.3, interpolate=True)
    plt.legend(loc='lower right', ncol=1, mode="expand")

    plt.savefig("ROCLensDetectNet_Full_%s.png" % TR)

def analyze_data(x_data, y_data, dataset_size, TR, shots, input_shape, meta_iters, version, normalize, step, phase):
    colors = ("r", "g", "b")
    channels = (0, 1, 2)
    fraction = 1
    percount = 0
    print("\n ** Generating data_analysis graph...")
    plt.figure()
    #if step == "Normalized" or step == "Resized & Normalized":
    #    plt.xlim([0,3])
    #else:
    plt.xlim([0,256])
    for z in range(int(dataset_size/fraction)):
        perc = int(z/int(dataset_size)*100)
        if perc == percount:
            print(" -- fraction done: %s percent" % perc)
            percount = percount + 1
        image = x_data[z]
        for channel, color in zip(channels, colors):
            histogram, bin_edges = np.histogram(image[:,:,channel], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=color)
    plt.xlabel("Color value")
    plt.ylabel("Pixels")
    plt.grid()
    plt.title("Color Histogram: %s-data, %s" % (phase, step))
    plt.savefig("Color_Hist_IMG_{}_step_{}_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(phase, step, TR, shots, input_shape, input_shape, meta_iters, version, normalize))

def ResNet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation_layer='relu', batch_normalization=True, conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    ** Arguments:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation_layer (string): activation_layer name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
    ** Returns
        x (tensor): tensor as input to the next layer"""

    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation_layer is not None:
            x = Activation(activation_layer)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation_layer is not None:
            x = Activation(activation_layer)(x)
        x = conv(x)
    return x

def ResNet_Generator(input_shape, depth, num_classes, inputs):
    """ResNet Version 1 Model builder [a]

    Arguments:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    Returns:
        model (Model): Keras model instance."""
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    #Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    x = ResNet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                strides = 2  # downsample
            y = ResNet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = ResNet_layer(inputs=y,
                             num_filters=num_filters,
                             activation_layer=None)
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                # linear projection residual shortcut connection to match
                # changed dims
                x = ResNet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation_layer=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
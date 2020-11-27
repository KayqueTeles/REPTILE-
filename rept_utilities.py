""" Utility functions. """
import numpy as np, os, random, shutil, sklearn, keras, wget, zipfile, tarfile, matplotlib.pyplot as plt, bisect, cv2

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers import Input
from collections import Counter

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

def fileremover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer):

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
    if os.path.exists('./model_REPTILE_version_%s.png' % version):
        os.remove('./model_REPTILE_version_%s.png' % version)
        piccounter = piccounter + 1
    if os.path.exists('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        os.remove('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
        piccounter = piccounter + 1    
    
    for lo in range(10):
        for b in range(TR):   
            if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, b)):
                os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, b))
                piccounter = piccounter + 1   

    print(" ** Removing done. %s files removed." % (piccounter))

    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling)):
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
    

def filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer):

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
    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
        print(" ** Done!")

    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling))
        print(" ** Done!")

    dest1 = ('/home/kayque/LENSLOAD/REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}'. format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer))
    dest2 = ('/home/kayque/LENSLOAD/REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}/SAMPLES'. format(TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer))

    if os.path.exists('./Code_data_version_%s.csv' % version):
        shutil.move('./Code_data_version_%s.csv' % version, dest1)
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
    if os.path.exists('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize)):
        shutil.move('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, version, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./model_REPTILE_version_%s.png' % version):
        shutil.move('./model_REPTILE_version_%s.png' % version, dest1)
        counter = counter + 1
    
    for lo in range(10):
        for b in range(TR):   
            if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, b)):
                shutil.move('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(TR, version, lo, input_shape, input_shape, b), dest2)
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
    print(" -- binary_cross step.")
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
    print(" -- binary_cross step.")
    for i in range(len(actual)):
        print(sum_square_error)
        sum_square_error += (actual[i] - predicted[i])**2.0
    mean_square_error = 1.0 / len(actual) * sum_square_error
    print(predicted[i])
    print(actual[i])
    return mean_square_error

####################################################3#############
##################################################################F
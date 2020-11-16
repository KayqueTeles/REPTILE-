""" Utility functions. """
import numpy as np, os, random, shutil, sklearn, keras, wget, zipfile, tarfile, matplotlib.pyplot as plt, bisect, cv2

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from collections import Counter

def fileremover(TR, version, shots, input_shape, meta_iters, normalize):

    piccounter = 0
    print('\n ** Removing specified files and folders...')
    if os.path.exists('./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize)):
        os.remove('./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize)):
        os.remove('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize))
        piccounter = piccounter + 1
    if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize)):
        os.remove('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize))
        piccounter = piccounter + 1   
    
    for lo in range(10):
        for b in range(TR*100):
            if os.path.exists("save_im_ver_%s_ch_%s_ind_%s.png" % (version, lo, b)):
                os.remove("save_im_ver_%s_ch_%s_ind_%s.png" % (version, lo, b))
                piccounter = piccounter + 1     
            if os.path.exists("save_im_full_ver_%s_ind_%s.png" % (version, b)):
                os.remove("save_im_full_ver_%s_ind_%s.png" % (version, b))
                piccounter = piccounter + 1     

    print(" ** Removing done. %s .png files removed." % (piccounter))

    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version)):
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
    

def filemover(TR, version, shots, input_shape, meta_iters, normalize):

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
    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
        print(" ** Done!")

    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}/SAMPLES". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
        print(" ** Done!")

    dest1 = ('/home/kayque/LENSLOAD/REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}'. format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
    dest2 = ('/home/kayque/LENSLOAD/REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}/SAMPLES'. format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))

    if os.path.exists('./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize)):
        shutil.move("./Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize)):
        shutil.move('./ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize), dest1)
        counter = counter + 1
    if os.path.exists('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize)):
        shutil.move('./EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png'. format(TR, shots, input_shape, input_shape, meta_iters, normalize), dest1)
        counter = counter + 1
    
    for lo in range(10):
        for b in range(TR*50): 
            if os.path.exists("save_im_ver_%s_ch_%s_ind_%s.png" % (version, lo, b)):
                shutil.move("save_im_ver_%s_ch_%s_ind_%s.png" % (version, lo, b), dest2)
                counter = counter + 1     
            if os.path.exists("save_im_full_ver_%s_ind_%s.png" % (version, b)):
                shutil.move("save_im_full_ver_%s_ind_%s.png" % (version, b), dest2)
                counter = counter + 1  
    print(" ** Moving done. %s files moved." % counter)
    print(dest1)

def ROCCurveCalculate(y_test, x_test, model):

    probs = model.predict(x_test)
    probsp = probs[:, 1]
    y_new = y_test#[:, 1]
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
    print('\n ** Checking files...')
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

def TestSamplesBalancer(y_data, x_data, vallim, TR, split):
    
    y_size = len(y_data)
    y_yes, y_no, y_excess = ([] for i in range(3))
    PARAM = TR/2
    print(' -- Applying classes balancing...')
    for y in range(0,y_size,1):
        if y_data[y] == 1:
            if len(y_yes)<(PARAM):
                y_yes = np.append(int(y), y_yes)
            else: 
                y_excess = np.append(int(y), y_excess)
        else:
            if len(y_no)<(PARAM):
                y_no = np.append(int(y), y_no)
            else: 
                y_excess = np.append(int(y), y_excess)
                
    y_y = np.append(y_no, y_yes)
    np.random.shuffle(y_y)

    np.random.shuffle(y_excess)
    y_y = y_y.astype(int)
    y_excess = y_excess.astype(int)

    if split == "train":
        y_data = y_data[y_y]
        x_data = x_data[y_y]
    else:
        y_data = y_data[y_excess]
        x_data = x_data[y_excess]
    print(" ** split:")
    print(split)

    print(" ** x_data:  ", x_data.shape)
    print(" ** y_data:  ", y_data.shape)

    return [y_data, x_data]

def FScoreCalc(y_test, x_test, model):

    probsp = np.argmax(model.predict(x_test), axis=-1)
    y_test = np.argmax(y_test, axis =-1)

    f_1_score = sklearn.metrics.f1_score(y_test, probsp)
    f_001_score = sklearn.metrics.fbeta_score(y_test, probsp, beta=0.01)
    
    print('\n ** F1_Score: %s, F0.01_Score: %s' % (f_1_score, f_001_score))
    return [f_1_score, f_001_score]

def imadjust(src, tol=0.5, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(int(src.shape[0])):
            for c in range(int(src.shape[1])):
                hist[int(src[r,c])] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    if (vin[1] - vin[0]) > 0:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    else:
        scale = 0
        
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

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
    #temp_image = np.stack((vector[:, :, 0],) * 3, axis=2)
    image = vector
    #for tt in range(3):
        
        #image[:,:,tt] = np.float32(image[:,:,tt])
        #image[:,:,tt] = cv2.normalize(image[:,:,tt], None, 0, 255, cv2.NORM_MINMAX)
        #image[:,:,tt] = np.uint8(image[:,:,tt])
        #image[:,:,tt] = imadjust(image[:,:,tt])
        #other = plt.figure()
        #plt.imshow(image[:,:,tt])
        #other.savefig("save_im_ver_%s_ch_%s_ind_%s.png" % (version, tt, index))
        #image[:,:,tt] = cv2.fastNlMeansDenoising(image[:,:,tt].astype(np.uint8),None,30,7,21)
    #image = np.rollaxis(image, 2, 0) 
    index = index + 1
    image = toimage(image)
    image.save("save_im_full_ver_%s_ind_%s.png" % (version, index))
    image = np.array(image)
    return [image, index]

def save_clue(vector, label, version, index, step, input_size, nrows, ncols):

    _, arrt = plt.subplots(nrows=rows, ncols=cols, figsize=(nrows, ncols))

    for a in range(nrows):
        for b in range(ncols):
            img = vector[index, :, :, :]
            img = np.stack((img[:, :, 0],) * 3, axis=2)
            img *= 255
            img = np.clip(img, 0, 255).astype("uint8")
            arrt[a, b].imshow(img)#, cmap="gray")
            arrt[a, b].xaxis.set_visible(False)
            arrt[a, b].yaxis.set_visible(False)
            index = index + 1
            
    plt.savefig("COLLECTION_step_{}_save_{}_{}x{}_size_{}th_version.png". format(step, 7, input_shape, input_shape, version))

    return [index]

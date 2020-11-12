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
from pathlib import Path

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')
##################################################

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
        for al in range(10):
            for b in range(TR*100):
               if os.path.exists("./temp_image_{}_{}_step_{}.png".     format(al,b,lo)):
                   os.remove("./temp_image_{}_{}_step_{}.png". format(al,b,lo))
                   piccounter = piccounter + 1
               if os.path.exists("./temp_image_{}_{}_step_train.png". format(al,b)):
                   os.remove("./temp_image_{}_{}_step_train.png". format(al,b))
                   piccounter = piccounter + 1     

    print(" ** Removing done. %s .png files removed." % (piccounter))

    if os.path.exists("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version)):
        shutil.rmtree("REPT-GRAPHS/REP_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}". format(TR, shots, input_shape, input_shape, meta_iters, normalize, version))
    
for i in range(10):
    print(' -- i = %s' % i)
    fileremover(16000, i, 10, 101, 2000, 'yes')
    fileremover(16000, i, 20, 101, 2000, 'yes')

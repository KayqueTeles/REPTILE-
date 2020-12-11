import numpy as np, os, random, shutil, sklearn, zipfile, tarfile, matplotlib.pyplot as plt
import bisect, cv2, tensorflow as tf, pandas as pd, h5py, time, csv, warnings
from tensorflow import keras

from keras.applications.resnet50 import ResNet50#, VGG16, InceptionV3
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.applications import InceptionV3

from tensorflow.keras import layers
from pathlib import Path
from PIL import Image
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss
from keras import backend as K
from keras.layers import Input
from collections import Counter
from data_generator01 import Dataset
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover
from rept_utilities import save_image, save_clue, conv_window, distrib_graph, class_choose
from rept_utilities import examples_graph, roc_curve_graph, acc_graph, loss_graph, FScoreCalc
from rept_utilities import basic_conv_model, ResNet_Sequential, analyze_data

warnings.filterwarnings("ignore")
Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

print("Num GPUs Available: ", len(tf.test.gpu_device_name()))
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.00001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 100    ####ORIGINALLY 25     -100
eval_batch_size = 100    ###ORIGINALLY 25   -100

meta_iters = 2007        #ORIGINALLY 2000    -5000
eval_iters = 5          ###ORIGINALLY 5     -20
inner_iters = 4            ##ORIGINALLY 4   -19
dataset_size = 20000
TR = int(dataset_size*0.6)
vallim = int(dataset_size*0.2)
version = 29
index = 0

eval_interval = 1
train_shots = 160        ##ORIGINALLY 20   -80
shots = 40             ###ORIGINALLY 5    -20
num_classes = 2   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 50  #originally 28 for omniglot  ###MUST BE AT LEAST 75 FOR INCEPTION
rows = 2
cols = 10
num_channels = 3
activation_layer = "relu"
output_layer = "sigmoid"    #original = softmax for REPTILE
normalize = 'BatchNormalization' #or 'none'
maxpooling = "yes"
dropout = 0.0
architecture = "ResNet50"
optimizer = "Adam"
classes = ['lens', 'not-lens']

print("\n\n\n ******** INITIALYZING CODE - REPTILE ********* \n ** Chosen parameters:")

code_data =[["learning rate", learning_rate],
            ["meta_step_size", meta_step_size],
            ["inner_batch_size", inner_batch_size],
            ["eval_batch_size", eval_batch_size], ["meta_iters", meta_iters],
            ["eval_iters", eval_iters], ["inner_iters", inner_iters],
            ["eval_interval", eval_interval], ["train_shots", train_shots],
            ["shots", shots], ["classes", num_classes],
            ["input_shape", input_shape],
            ["dataset_size", dataset_size], ["TR", TR], ["valid", vallim],
            ["num_channels", num_channels],
            ["architecture", architecture],
            ["activation_layer", activation_layer],
            ["output_layer", output_layer],
            ["optimizer", optimizer],
            ["normalization", normalize],["maxpooling", maxpooling],
            ["dropout", dropout], ["VERSION", version]]
print(code_data)
###CLEAN UP PREVIOUS FILES
fileremover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling, architecture, learning_rate)
###DOWNLOAD DATASET
data_downloader() 
###SAVE_CSV WITH CODE DATA.
with open('Code_data_version_{}_learning_rate_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}_version_{}_activation_layer_{}_output_layer_{}_maxpooling_{}_arch_{}.csv'. format(learning_rate, TR, shots, input_shape, input_shape, meta_iters, normalize, version, activation_layer, output_layer, maxpooling, architecture), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(code_data)

###IMPORT DATASET TO CODE
path = os.getcwd() + "/" + "lensdata/"
labels = pd.read_csv(path + 'y_data20000fits.csv',delimiter=',', header=None)
y_data = np.array(labels, np.uint8)
x_datasaved = h5py.File(path + 'x_data20000fits.h5', 'r')
x_datas = x_datasaved['data']
x_datas = x_datas[:,:,:,0:num_channels]

print(" ** Shuffling data...")
y_vec = np.array([i for i in range(int(len(y_data)))])
np.random.shuffle(y_vec)
y_data = y_data[y_vec]
x_datas = x_datas[y_vec]

analyze_data(x_datas, y_data, dataset_size, TR, shots, input_shape, meta_iters, version, normalize, "Before Norm", "train")

index = save_clue(x_datas, y_data, TR, version, 1, input_shape, 5, 5, index)
#x_datas = (x_datas - np.nanmin(x_datas))/np.ptp(x_datas)
x_datas/255.0

analyze_data(x_datas, y_data, dataset_size, TR, shots, input_shape, meta_iters, version, normalize, "Normalized", "train")

print(x_datas.shape)
x_data = tf.image.resize(x_datas, [input_shape, input_shape], preserve_aspect_ratio=True)
x_data = np.array(x_data)
print(x_data.shape)

#analyze_data(x_data, y_data, dataset_size, TR, shots, input_shape, meta_iters, version, normalize, "Resized & Normalized", "train")

x_test = x_data[(TR+vallim):int(len(y_data)),:,:,:]
y_test = y_data[(TR+vallim):int(len(y_data))]

#analyze_data(x_test, y_test, dataset_size, TR, shots, input_shape, meta_iters, version, normalize, "Resized & Normalized", "test")

distrib_graph(y_data[0:TR], y_data[TR:(TR+vallim)], y_data[(TR+vallim):int(len(y_data))], classes, TR)

print(" ** Network building stage...")

img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
img_input = Input(shape=img_shape)
if architecture == "ResNet50":
    model = ResNet50(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
elif architecture == "VGG16":
    model = VGG16(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
    x_data = preprocess_input(x_data)
elif architecture == "InceptionV3":
    model = InceptionV3(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
else:
    raise Exception(" Please make sure you have typed network-type correctly.")

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model.summary()
model.compile(loss= 'binary_crossentropy', optimizer=optimizer, run_eagerly=True)
plot_model(model,  to_file="model_REPTILE_version_%s.png" % version)

filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling, architecture, learning_rate)

print('\n ** Mission accomplished.')
print("\n ** FINISHED! ************************")
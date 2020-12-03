import numpy as np, os, random, shutil, sklearn, wget, zipfile, tarfile, matplotlib.pyplot as plt
import bisect, cv2, tensorflow as tf, pandas as pd, h5py, time, csv
from tensorflow import keras
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
from data_generator04 import Dataset
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover, FScoreCalc
from rept_utilities import save_image, save_clue, conv_window, miniimagenet_downloader
from rept_utilities import examples_graph, roc_curve_graph, acc_graph, loss_graph

Path('/home/kayque/MINIIMAGELOAD/').parent
os.chdir('/home/kayque/MINIIMAGELOAD/')
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 25    ####ORIGINALLY 25     -100
eval_batch_size = 25    ###ORIGINALLY 25   -100

meta_iters = 2000        #ORIGINALLY 2000    -5000
eval_iters = 5          ###ORIGINALLY 5     -20
inner_iters = 4            ##ORIGINALLY 4   -19
dataset_size = 20000
TR = int(dataset_size*0.8)
vallim = int(dataset_size*0.2)
version = 25 
index = 0

eval_interval = 1
train_shots = 20        ##ORIGINALLY 20   -80
shots = 5             ###ORIGINALLY 5    -20
num_classes = 10   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 84  #originally 28 for omniglot
rows = 2
cols = 10
num_channels = 1
activation_layer = "relu"
output_layer = "sigmoid"    #original = softmax for REPTILE
normalize = 'batch_normalization' #or 'none'
maxpooling = "no"
dropout = 0.1

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
            ["activation_layer", activation_layer],
            ["output_layer", output_layer],
            ["normalization", normalize],["maxpooling", maxpooling],
            ["dropout", dropout], ["VERSION", version]]
print(code_data)

###CLEAN UP PREVIOUS FILES
fileremover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling)
###DOWNLOAD DATASET
miniimagenet_downloader() 

###SAVE_CSV WITH CODE DATA.
with open('Code_data_version_%s.csv' % version, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(code_data)

print("\n ** Building dataset functions:")
print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(split="train", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(split="test", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(test_dataset)
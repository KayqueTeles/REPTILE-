import numpy as np, os, random, shutil, sklearn, wget, zipfile, tarfile, matplotlib.pyplot as plt
import bisect, cv2, tensorflow as tf, pandas as pd, h5py, time, csv
from tensorflow import keras
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from pathlib import Path
from PIL import Image
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss
from keras import backend as K
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from collections import Counter
from data_generator01 import Dataset
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover
from rept_utilities import save_image, save_clue, conv_window
from rept_utilities import examples_graph, roc_curve_graph, acc_graph, loss_graph, FScoreCalc

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.01 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 25    ####ORIGINALLY 25     -100
eval_batch_size = 25    ###ORIGINALLY 25   -100

meta_iters = 2000        #ORIGINALLY 2000    -5000
eval_iters = 5          ###ORIGINALLY 5     -20
inner_iters = 4            ##ORIGINALLY 4   -19
dataset_size = 20000
TR = int(dataset_size*0.8)
vallim = int(dataset_size*0.2)
version = 29
index = 0

eval_interval = 1
train_shots = 20        ##ORIGINALLY 20   -80
shots = 5             ###ORIGINALLY 5    -20
num_classes = 2   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 101  #originally 28 for omniglot
rows = 2
cols = 10
num_channels = 3
activation_layer = "relu"
output_layer = "softmax"    #original = softmax for REPTILE
normalize = 'BatchNormalization' #or 'none'
maxpooling = "yes"
dropout = 0.0
architecture = "ResNet"

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
            ["normalization", normalize],["maxpooling", maxpooling],
            ["dropout", dropout], ["VERSION", version]]
print(code_data)
filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling)
###CLEAN UP PREVIOUS FILES
fileremover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling)
###DOWNLOAD DATASET
data_downloader() 
###SAVE_CSV WITH CODE DATA.
with open('Code_data_version_%s.csv' % version, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(code_data)

###IMPORT DATASET TO CODE
path = os.getcwd() + "/" + "lensdata/"
labels = pd.read_csv(path + 'y_data20000fits.csv',delimiter=',', header=None)
y_data = np.array(labels, np.uint8)
x_datasaved = h5py.File(path + 'x_data20000fits.h5', 'r')
x_data = x_datasaved['data']
x_data = x_data[:,:,:,0:num_channels]

index = save_clue(x_data, y_data, TR, version, 1, input_shape, 5, 5, index)
x_data = (x_data - np.nanmin(x_data))/np.ptp(x_data)

x_data = np.moveaxis(x_data, -1, 0) 
x_data = np.moveaxis(x_data, -3, 0)

x_test = x_data[TR:int(len(y_data)),:,:,:]
y_test = y_data[TR:int(len(y_data))]

print("\n ** Building dataset functions:")
print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(x_data=x_data, y_data=y_data, split="train", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(x_data=x_data, y_data=y_data, split="test", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(test_dataset)

examples_graph(rows, cols, train_dataset, index, TR, shots, input_shape, meta_iters, version, normalize)

print(" ** Network building stage...")

#ORIGINAL: KERNEL_SIZE = 3, STRIDES = 2, PADDING = "SAME", FILTERS = 64
model = Sequential()

model.add(Convolution2D(64, (3, 3), input_shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]), data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3, 3), data_format='channels_first')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3, 3), data_format='channels_first')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3, 3), data_format='channels_first')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.5)) # 0.2
model.add(Flatten())

model.add(Dense(512)) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256)) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64)) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Dense(2, activation= 'softmax' )) 
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(loss= 'binary_crossentropy' , optimizer=optimizer, metrics=[ 'accuracy' ])
model.summary()
plot_model(model,  to_file="model_REPTILE_version_%s.png" % version)

print(" ** Network successfully built.")
print("\n ** INITIALYZING REPTILE NETWORK.")
begin = time.perf_counter()

count = 5

try:
    training, testing, tes_losses, tra_losses = ([] for i in range(4))
    print("\n ** IT'S TIME TO META_TRAIN!")
    for meta_iter in range(meta_iters):   ##FROM 0 TO 2000
        meta_step_timer = time.perf_counter()
        frac_done = meta_iter / meta_iters
        print('\n **********************************\n **** STEP: {}/{} \n ** Fraction done: {} %.'. format((meta_iter+1), meta_iters, frac_done))
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        old_vars = model.get_weights()  # Temporarily save the weights from the model.
        mini_dataset = train_dataset.get_mini_dataset(
            inner_batch_size, inner_iters, train_shots,
            num_classes, num_channels)   # Get a sample from the full dataset.
        #lab = blind_dataset.get_mini_dataset(
          #          inner_batch_size, inner_iters, train_shots, num_classes, num_channels)
        #print(lab)
        print(" -- Clue of mini_dataset: ", mini_dataset)  ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
        cycle = 1
        for images, labels in mini_dataset:
            with tf.GradientTape() as tape:
                preds = model(images)
                loss = keras.losses.sparse_categorical_crossentropy(labels, preds) 
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(" -- images.shape from trainset cycle %s: %s" % (cycle, images.shape))
            print(" -- labels.shape from trainset cycle %s: %s" % (cycle, labels.shape))
            cycle = cycle + 1
        new_vars = model.get_weights()
        #new_vars = model.load_weights("model.h5")
        # Perform SGD for the meta step.
        for var in range(len(new_vars)):
            new_vars[var] = old_vars[var] + (
                (new_vars[var] - old_vars[var]) * cur_meta_step_size
            )
        # After the meta-learning step, reload the newly-trained weights into the model.
        model.set_weights(new_vars)
        print("\n ** EVALUATION LOOP!")
        # Evaluation loop
        if meta_iter % eval_interval == 0:
            accuracies, mean_loss = ([] for i in range(2))
            for dataset in (train_dataset, test_dataset):
                hg_loss, mn_loss, lw_loss = ([] for i in range(3))
                # Sample a mini dataset from the full dataset.
                train_set, test_images, test_labels = dataset.get_mini_dataset(
                    eval_batch_size, eval_iters, shots, num_classes, num_channels, split="training"
                )
                print(" -- TRAINSET:", train_set)   ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
                #print(" -- TESTIMAGES:")
                #print(test_images)  ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
                print(" -- TESTLABELS:", test_labels)    ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
                old_vars = model.get_weights()
                # Train on the samples and get the resulting accuracies.
                cycle = 1
                for images, labels in train_set:
                    print(" -- images.shape from trainset cycle %s: %s" % (cycle, images.shape))
                    print(" -- labels.shape from trainset cycle %s: %s" % (cycle, labels.shape))
                    cycle = cycle + 1
                    index = index + 1
                    #index = save_clue(images, labels, TR, version, 7, input_shape, 3, 3, index)
                    print(images[0].shape)
                    with tf.GradientTape() as tape:
                        preds = model(images)
                        #print("preds: ", preds)
                        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                        #print("loss: ", loss)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                test_preds = model.predict(test_images)
                test_preds = tf.argmax(test_preds).numpy()
                num_correct = (test_preds == test_labels).sum()
                mean_loss.append(log_loss(test_labels, test_preds))
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / num_classes)
                #tpr, fpr, auc, auc2, thres = ROCCurveCalculate(test_labels, test_images, model)
                #roc_curve_graph(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, index)
            tra_losses.append(mean_loss[0])
            tes_losses.append(mean_loss[1])
            training.append(accuracies[0])
            testing.append(accuracies[1])
            if meta_iter % 100 == 0:
                print(
                    " ** batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
                )
            elapsed = int((time.perf_counter() - meta_step_timer))
            print(" ** step_time: %s seconds." % elapsed)

    train_y = conv_window(training)
    test_y = conv_window(testing)
    tra_loss = conv_window(tra_losses)
    tes_loss = conv_window(tes_losses)

    acc_graph(test_y, train_y, TR, shots, input_shape, meta_iters, version, normalize)
    loss_graph(tra_loss, tes_loss, TR, shots, input_shape, meta_iters, version, normalize)

    test_preds = model.predict(test_images)
    tpr, fpr, auc, auc2, thres = ROCCurveCalculate(y_test, x_test, model)
    roc_curve_graph(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, index)

    #f1_score, f001_score = FScoreCalc(test_labels, test_images, model)

    #writer.writerows([["f1", f1_score],
    #                 ["f001", f001_score]])

except AssertionError as error:
    print(error)
#except:
    #print("\n\n\n ************ WARNING *********** \n\n ** Something went wrong during execution.\
     #        Please check it out later. ** Proceeding to move generated files...")
    #filemover(TR, version, shots, input_shape, meta_iters, normalize, output_layer)
    pass

filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling)

timee = int((time.perf_counter() - begin)/(60))
print('\n ** Mission accomplished in %s minutes.' % timee)
print("\n ** FINISHED! ************************")
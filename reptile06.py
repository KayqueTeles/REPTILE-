import numpy as np, os, random, shutil, sklearn, wget, zipfile, tarfile, matplotlib.pyplot as plt
import bisect, cv2, tensorflow as tf, pandas as pd, h5py, time, csv
from tensorflow import keras
from keras.applications.resnet50 import ResNet50#, VGG16, InceptionV3
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
from data_generator06 import Dataset
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover
from rept_utilities import save_image, save_clue, conv_window, distrib_graph, extraction, basic_conv_model
from rept_utilities import examples_graph, roc_curve_graph, acc_graph, loss_graph, FScoreCalc, multiclass_roc_graphs
import tensorflow_datasets as tfds

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')
print(tf.executing_eagerly())
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 100    ####ORIGINALLY 25     -100
eval_batch_size = 100    ###ORIGINALLY 25   -100

meta_iters = 2001        #ORIGINALLY 2000    -5000
eval_iters = 5          ###ORIGINALLY 5     -20
inner_iters = 4            ##ORIGINALLY 4   -19
dataset_size = 20000
TR = int(dataset_size*0.6)
vallim = int(dataset_size*0.2)
version = 29
index = 0

eval_interval = 1
train_shots = 80        ##ORIGINALLY 20   -80
shots = 20             ###ORIGINALLY 5    -20
num_classes = 5   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 56  #originally 28 for omniglot
rows = 2
cols = 10
num_channels = 1
activation_layer = "relu"
output_layer = "softmax"    #original = softmax for REPTILE
normalize = 'BatchNormalization' #or 'none'
maxpooling = "yes"
dropout = 0.0
architecture = "TrueResNet50-Omniglot"
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
            ["normalization", normalize],["maxpooling", maxpooling],
            ["dropout", dropout], ["VERSION", version]]
print(code_data)
###CLEAN UP PREVIOUS FILES
fileremover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling, architecture, learning_rate)
###DOWNLOAD DATASET
#data_downloader() 
###SAVE_CSV WITH CODE DATA.
with open('Code_data_version_%s.csv' % version, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(code_data)

###IMPORT DATASET TO CODE
ds1 = tfds.load("omniglot", split="test", as_supervised=True, shuffle_files=False)
x_test = np.zeros(shape =(13180, input_shape, input_shape, 1))
y_test = []

print("** Gathering test_set...")
for image, label in ds1.map(extraction):
    image = image.numpy()
    image = tf.image.resize(image, [input_shape, input_shape], preserve_aspect_ratio=True)
    image = np.array(image)
    print(image.shape)
    label = str(label.numpy())
    x_test[int(label)] = image
    y_test = np.append(y_test, int(label))

print("\n ** Building dataset functions:")
print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(split="train", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(split="test", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(test_dataset)

#examples_graph(rows, cols, train_dataset, index, TR, shots, input_shape, meta_iters, version, normalize)

print(" ** Network building stage...")

img_shape = (x_test.shape[1], x_test.shape[2], x_test.shape[3])
img_input = Input(shape=img_shape)
model = ResNet50(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model.summary()
model.compile(loss= 'categorical_crossentropy', optimizer=optimizer, run_eagerly=True)
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
            num_classes, input_shape, num_channels)   # Get a sample from the full dataset.
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
                    eval_batch_size, eval_iters, shots, num_classes, input_shape, num_channels, split="training"
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
                test_preds = model.predict_generator(test_images)
                print(test_preds, test_labels)
                test_preds = tf.argmax(test_preds).numpy()
                print(test_preds, test_labels)
                num_correct = (test_preds == test_labels).sum()
                print(test_preds, test_labels)
                mean_loss.append(log_loss(test_labels, test_preds))
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / num_classes)
                #tpr, fpr, auc, auc2, thres = ROCCurveCalculate(test_labels, test_images, model)
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
    multiclass_roc_graphs(num_classes, y_test, x_test, model, TR, shots, input_shape, meta_iters, version, normalize)

    f1_score, f001_score = FScoreCalc(y_test, x_test, model)
    #writer.writerows([["f1", f1_score],
                    # ["f001", f001_score]])

except AssertionError as error:
    print(error)
#except:
    #print("\n\n\n ************ WARNING *********** \n\n ** Something went wrong during execution.\
     #        Please check it out later. ** Proceeding to move generated files...")
    #filemover(TR, version, shots, input_shape, meta_iters, normalize, output_layer)
    pass

filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling, architecture, learning_rate)

timee = int((time.perf_counter() - begin)/(60))
print('\n ** Mission accomplished in %s minutes.' % timee)
print("\n ** FINISHED! ************************")

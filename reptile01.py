import numpy as np, os, random, shutil, sklearn, zipfile, tarfile, matplotlib.pyplot as plt
import bisect, cv2, tensorflow as tf, pandas as pd, h5py, time, csv, warnings
from tensorflow import keras

from keras.applications.resnet50 import ResNet50 #, VGG16, InceptionV3
from keras.applications import ResNet101
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
from rept_utilities import basic_conv_model, ResNet_Generator

warnings.filterwarnings("ignore")
Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

print("Num GPUs Available: ", len(tf.test.gpu_device_name()))
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.0001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 25    ####ORIGINALLY 25     -100
eval_batch_size = 25    ###ORIGINALLY 25   -100

meta_iters = 2000        #ORIGINALLY 2000    -5000
eval_iters = 5          ###ORIGINALLY 5     -20
inner_iters = 4            ##ORIGINALLY 4   -19
dataset_size = 20000
TR = int(dataset_size*0.8)
vallim = int(dataset_size*0.1)
version = 30
index = 0

eval_interval = 1
train_shots = 20        ##ORIGINALLY 20   -80
shots = 5             ###ORIGINALLY 5    -20
num_classes = 2   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 101  #originally 28 for omniglot  ###MUST BE AT LEAST 75 FOR INCEPTION
rows = 2        ##
cols = 10
num_channels = 3
activation_layer = "relu"
output_layer = "sigmoid"    #original = softmax for REPTILE
normalize = 'BatchNormalization' #or 'none'
maxpooling = "noney"
dropout = 0.5
architecture = "ResNet101"

optimizer = "SGD"
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

index = save_clue(x_datas, y_data, TR, version, 1, input_shape, 5, 5, index)
#x_datas = (x_datas - np.nanmin(x_datas))/np.ptp(x_datas)

print(x_datas.shape)
x_data = tf.image.resize(x_datas, [input_shape, input_shape], preserve_aspect_ratio=True)
x_data = np.array(x_data)
print(x_data.shape)

x_test = x_data[(TR+vallim):int(len(y_data)),:,:,:]
y_test = y_data[(TR+vallim):int(len(y_data))]

distrib_graph(y_data[0:TR], y_data[TR:(TR+vallim)], y_data[(TR+vallim):int(len(y_data))], classes, TR)

print("\n ** Building dataset functions:")
print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(x_data=x_data, y_data=y_data, split="train", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(x_data=x_data, y_data=y_data, split="test", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(test_dataset)

print(" ** Network building stage...")
img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
img_input = Input(shape=img_shape)
if architecture == "ResNet50":
    model = ResNet50(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
elif architecture == "VGG16":
    model = VGG16(include_top=False, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
    x_data = preprocess_input(x_data)
elif architecture == "InceptionV3":
    model = InceptionV3(include_top=False, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
elif architecture == "Basic":
    model = basic_conv_model(normalize, dropout, maxpooling, activation_layer, output_layer, input_shape, num_channels, 128, 3, "same", learning_rate, optimizer, num_classes)
elif architecture == "ResNet":
    model = ResNet_Generator(input_shape, 32, num_classes, img_input)
elif architecture == "ResNet101":
    model = ResNet101(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
else:
    raise Exception(" Please make sure you have typed network-type correctly.")

optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, run_eagerly=True)
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
                num_correct = (test_preds == test_labels).sum() #sum(1 for a, b in zip(test_preds, test_labels) if a == b[0])
                mean_loss.append(log_loss(test_labels, test_preds))
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / num_classes)
                #tpr, fpr, auc, auc2, thres = ROCCurveCalculate(test_labels, test_images, model)
            tra_losses.append(mean_loss[0])
            tes_losses.append(mean_loss[1])
            training.append(accuracies[0])
            testing.append(accuracies[1])
            print("train_acc:", training)
            print("test_acc:", testing)
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
    f1_score, f001_score = FScoreCalc(y_test, x_test, model)
    roc_curve_graph(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, f1_score, f001_score)

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
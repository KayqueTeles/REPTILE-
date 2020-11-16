import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds   ###voa english
from data_generator01 import Dataset
from pathlib import Path
import os
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover, save_image, save_clue
from PIL import Image

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000        #ORIGINALLY 2000
eval_iters = 5
inner_iters = 4
dataset_size = 20000
TR = int(dataset_size*0.8)
vallim = int(dataset_size*0.2)
version = 9  
index = 0
normalize = 'yes'

eval_interval = 1
train_shots = 20
shots = 10
num_classes = 2   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 101  #originally 101 for omnigloto
rows = 2
cols = 10
num_channels = 3
####################################################3
print("\n\n\n ******** INITIATING REPTILE NETWORK ********* \n ** Chosen parameters: \n -- learning rate: %s; \n -- meta_step_size: %s; \n -- inner_batch_size: %s; \n -- eval_batch_size: %s; \n -- meta_iters: %s; \n eval_iters: %s; \n -- inner_iters: %s; \n -- eval_interval: %s; \n -- train_shots: %s; \n -- shots: %s, \n -- classes: %s; \n -- input_shape: %s; \n -- rows: %s; \n -- cols: %s; \n -- num_channels: %s; \n\n\n -- VERSION: %s." % (learning_rate, meta_step_size, inner_batch_size, eval_batch_size, meta_iters, eval_iters, inner_iters, eval_interval, train_shots, shots, num_classes, input_shape, rows, cols, num_channels, version))


fileremover(TR, version, shots, input_shape, meta_iters, normalize)

#import urllib3
#urllib3.disable_warnings()  # Disable SSL warnings that may happen during download.
print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(training=True, version=version, TR=TR, vallim=vallim, index=index, input_shape=input_shape)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(training=False, version=version, TR=TR, vallim=vallim, index=index, input_shape=input_shape)
print(test_dataset)

data_downloader()  #DO NOT FORGET TO USE PIP INSTALL WGET

_, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))

sample_keys = list(train_dataset.data.keys())
    

for a in range(rows):
    for b in range(cols):
        temp_image = train_dataset.data[sample_keys[a]][b]
        print(temp_image.shape)
        #temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
        #temp_image = np.stack(temp_image, axis=2)
        temp_image = toimage(temp_image)
        #print(temp_image.shape)
        #temp_image *= 255
        #temp_image = np.clip(temp_image, 0, 255).astype("uint32")
        #print(temp_image.shape)
        if b == 2:
            axarr[a, b].set_title("Class : " + sample_keys[a])
        #imgs, index = save_image(temp_image, version, index, 3, input_shape)
        axarr[a, b].imshow(temp_image)#, cmap="gray")
        axarr[a, b].xaxis.set_visible(False)
        axarr[a, b].yaxis.set_visible(False)
plt.show()
plt.savefig("EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, normalize))

print(" ** Network building stage...")
def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    ##########DROPOUT: ADIÇÃO NOSSA! ON VERSION 4
    #x = layers.Dropout(0.1)(x)
    return layers.ReLU()(x)

inputs = layers.Input(shape=(input_shape, input_shape, 3))
x = conv_bn(inputs)
x = conv_bn(x)
x = conv_bn(x)
x = conv_bn(x)
x = layers.Flatten()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile()
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

print(" ** Network successfully built.")

training = []
testing = []
print('\n ** Meta-train step.')
for meta_iter in range(meta_iters):   ##FROM 0 TO 2000
    frac_done = meta_iter / meta_iters
    print(' ** Fraction done: {} %.'. format(frac_done))
    cur_meta_step_size = (1 - frac_done) * meta_step_size
    # Temporarily save the weights from the model.
    old_vars = model.get_weights()
    # Get a sample from the full dataset.
    mini_dataset = train_dataset.get_mini_dataset(
        inner_batch_size, inner_iters, train_shots,
        num_classes)
    
    counter = 0
    print(" -- MINIDATASET:")
    print(mini_dataset)  ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
    for images, labels in mini_dataset:
        #print(' -- cycle: %s. ' % counter)
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
        #print(' -- proceeding to gradient steps..')
        #for x in range(len(labels)):
        #    img = images[x]
        #    img, index = save_minidataset(img, version, index, 4, input_shape)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        counter = counter + 1
    #print(' -- somewhat done.')
    new_vars = model.get_weights()
    #new_vars = model.load_weights("model.h5")
    # Perform SGD for the meta step.
    for var in range(len(new_vars)):
        new_vars[var] = old_vars[var] + (
            (new_vars[var] - old_vars[var]) * cur_meta_step_size
        )
    # After the meta-learning step, reload the newly-trained weights into the model.
    model.set_weights(new_vars)
    print(" ** AFTER META-LEARNING STEP!")
    # Evaluation loop
    if meta_iter % eval_interval == 0:
        accuracies = []
        for dataset in (train_dataset, test_dataset):
            # Sample a mini dataset from the full dataset.
            train_set, test_images, test_labels = dataset.get_mini_dataset(
                eval_batch_size, eval_iters, shots, num_classes, split=True
            )
            print(" -- TRAINSET:")
            print(train_set)   ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
            #print(" -- TESTIMAGES:")
            #print(test_images)  ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
            print(" -- TESTLABELS:")
            print(test_labels)    ##REPEAT_DATASET HAS NO ATTRIBUTE "SHAPE"
            old_vars = model.get_weights()
            # Train on the samples and get the resulting accuracies.
            for images, labels in train_set:
                print(" -- images.shape from trainset:")
                print(images.shape)
                print(" -- labels.shape from trainset:")
                print(labels.shape)
                with tf.GradientTape() as tape:
                    preds = model(images)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #for x in range(len(labels)):
            #    img = images[x]
            #    img, index = save_minidataset(img, version, index, 'train', input_shape)
            test_preds = model.predict(test_images)
            test_preds = tf.argmax(test_preds).numpy()
            num_correct = (test_preds == test_labels).sum()
            # Reset the weights after getting the evaluation accuracies.
            model.set_weights(old_vars)
            accuracies.append(num_correct / num_classes)
        training.append(accuracies[0])
        testing.append(accuracies[1])
        if meta_iter % 100 == 0:
            print(
                "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
            )

window_length = 100   #ORIGINALLY 100
train_s = np.r_[
    training[window_length - 1 : 0 : -1], training, training[-1:-window_length:-1]
]
test_s = np.r_[
    testing[window_length - 1 : 0 : -1], testing, testing[-1:-window_length:-1]
]
w = np.hamming(window_length)
train_y = np.convolve(w / w.sum(), train_s, mode="valid")
test_y = np.convolve(w / w.sum(), test_s, mode="valid")

# Display the training accuracies.
plt.figure()
x = np.arange(0, len(test_y), 1)
plt.plot(x, test_y, x, train_y)
plt.legend(["test", "train"])
plt.savefig("Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, normalize))
plt.grid()

train_set, test_images, test_labels = dataset.get_mini_dataset(
    eval_batch_size, eval_iters, shots, num_classes, split=True
)
for images, labels in train_set:
    with tf.GradientTape() as tape:
        preds = model(images)
        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
test_preds = model.predict(test_images)
tpr, fpr, auc, auc2, thres = ROCCurveCalculate(test_labels, test_images, model)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--') # k = color black
plt.plot(fpr, tpr, label="AUC: %.3f" % auc, linewidth=3) # for color 'C'+str(j), for j[0 9]
plt.legend(loc='lower right', ncol=1, mode="expand")
plt.title('ROC for %s training samples' % (TR))
plt.xlabel('false positive rate', fontsize=14)
plt.ylabel('true positive rate', fontsize=14)
    
plt.savefig("ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, normalize))
test_preds = tf.argmax(test_preds).numpy()

_, axarr = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))

sample_keys = list(train_dataset.data.keys())

filemover(TR, version, shots, input_shape, meta_iters, normalize)

#for i, ax in zip(range(5), axarr):
#    temp_image = np.stack((test_images[i, :, :, 0],) * 3, axis=2)
#    temp_image *= 255
#    temp_image = np.clip(temp_image, 0, 255).astype("uint8")
#    ax.set_title(
#        "Label : {}, Prediction : #{}".format(int(test_labels[i]), test_preds[i])
#    )
#    ax.imshow(temp_image, cmap="gray")
#    ax.xaxis.set_visible(False)
#    ax.yaxis.set_visible(False)
#plt.show()

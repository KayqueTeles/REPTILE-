import numpy as np, os, random, shutil, sklearn, wget, zipfile, tarfile, matplotlib.pyplot as plt
import bisect, cv2, tensorflow as tf, pandas as pd, h5py, time
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from PIL import Image
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers import Input
from collections import Counter
from data_generator01 import Dataset
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover, save_image, save_clue, binary_cross_entropy, mean_squared_error

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 100    ####ORIGINALLY 25
eval_batch_size = 100    ###ORIGINALLY 25

meta_iters = 5000        #ORIGINALLY 2000
eval_iters = 20          ###ORIGINALLY 5
inner_iters = 19            ##ORIGINALLY 4
dataset_size = 20000
TR = int(dataset_size*0.8)
vallim = int(dataset_size*0.1)
version = 21 ##VERSION 10: IMAGES STACKED INTO A SINGLE ONE  
index = 0
normalize = 'yes'

eval_interval = 1
train_shots = 80        ##ORIGINALLY 20
shots = 20             ###ORIGINALLY 5
num_classes = 2   #ORIGINALLY 5 FOR OMNIGLOT DATASET
input_shape = 101  #originally 28 for omniglot
rows = 2
cols = 10
num_channels = 3

print("\n\n\n ******** INITIALYZING CODE - REPTILE ********* \n ** Chosen parameters: \n -- learning rate: %s; \n -- meta_step_size: %s; \n -- inner_batch_size: %s; \n -- eval_batch_size: %s; \n -- meta_iters: %s; \n -- eval_iters: %s; \n -- inner_iters: %s; \n -- eval_interval: %s; \n -- train_shots: %s; \n -- shots: %s, \n -- classes: %s; \n -- input_shape: %s; \n -- rows: %s; \n -- cols: %s; \n -- num_channels: %s; \n -- VERSION: %s." % (learning_rate, meta_step_size, inner_batch_size, eval_batch_size, meta_iters, eval_iters, inner_iters, eval_interval, train_shots, shots, num_classes, input_shape, rows, cols, num_channels, version))

###CLEAN UP PREVIOUS FILES
fileremover(TR, version, shots, input_shape, meta_iters, normalize)
###DOWNLOAD DATASET
data_downloader() 

###IMPORT DATASET TO CODE
path = os.getcwd() + "/" + "lensdata/"
labels = pd.read_csv(path + 'y_data20000fits.csv',delimiter=',', header=None)
y_data = np.array(labels, np.uint8)
x_datasaved = h5py.File(path + 'x_data20000fits.h5', 'r')
x_data = x_datasaved['data']
x_data = x_data[:,:,:,0:num_channels]

index = save_clue(x_data, y_data, TR, version, 1, input_shape, 5, 5, index)
x_data = (x_data - np.nanmin(x_data))/np.ptp(x_data)

print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(x_data=x_data, y_data=y_data, split="train", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, 
num_channels=num_channels)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(x_data=x_data, y_data=y_data, split="test", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape,
num_channels=num_channels)
print(test_dataset)
print(" ** Blind_dataset is bein imported...")
blind_dataset = Dataset(x_data=x_data, y_data=y_data, split="blind", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape,
num_channels=num_channels)
print(test_dataset)

_, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
sample_keys = list(train_dataset.data.keys())

index = save_clue(x_data, y_data, TR, version, 5, input_shape, 5, 5, index)

for a in range(rows):
    for b in range(cols):
        temp_image = train_dataset.data[sample_keys[a]][b]
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
plt.savefig("EXAMPLE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))

print(" ** Network building stage...")
##kernel original: 3
def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    ##########DROPOUT: ADIÇÃO NOSSA! ON VERSION 4
    #x = layers.Dropout(0.1)(x)
    return layers.ReLU()(x)

inputs = layers.Input(shape=(input_shape, input_shape, num_channels))
x = conv_bn(inputs)
x = conv_bn(x)
x = conv_bn(x)
x = conv_bn(x)
x = layers.Flatten()(x)
outputs = layers.Dense(num_classes, activation=
"softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
##model.compile()  <-- ORIGINAL!
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(loss= 'binary_crossentropy', optimizer=optimizer , metrics=[ 'accuracy' ])

#plot_model(model,  to_file="model_REPTILE_version_%s.png" % version)

print(" ** Network successfully built.")
print("\n ** INITIALYZING REPTILE NETWORK.")
begin = time.perf_counter()

count = 5

try:
    training, testing, losses, train_losses, losses_mse = ([] for i in range(5))
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
                #print("preds: ", preds)
                #loss = keras.losses.binary_crossentropy(labels, preds) 
                loss = keras.losses.sparse_categorical_crossentropy(labels, preds) 
                #print("loss: ", loss)
            #index = save_clue(images, labels, TR, version, count, input_shape, 5, 5, index)
            #count = count + 1
            #print(' -- proceeding to gradient steps..')
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
            accuracies = []
            for dataset in (train_dataset, test_dataset):
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
                    with tf.GradientTape() as tape:
                        #preds = model.predict(images)
                        preds = model(images)
                        print("preds: ", preds)
                        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                        print("loss: ", loss)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    #index = save_clue(images, labels, TR, version, count, input_shape, 5, 5, index)
                    #count = count + 1
                test_preds = model.predict(test_images)
                test_preds = tf.argmax(test_preds).numpy()
                num_correct = (test_preds == test_labels).sum()
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / num_classes)
                print(" ** Loss.")
                mean_losses = binary_cross_entropy(test_labels, test_preds)
                print(mean_losses)
                losses.append(mean_losses)
                mean_loss_mse = mean_squared_error(test_labels, test_preds)
                print(mean_loss_mse)
                losses_mse.append(mean_loss_mse)
                #train_preds = model.predict()
                #mean_train_losses = binary_cross_entropy(labels, train_preds)
                #train_losses.append(mean_train_losses)
            training.append(accuracies[0])
            testing.append(accuracies[1])
            if meta_iter % 100 == 0:
                print(
                    " ** batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
                )
            elapsed = int((time.perf_counter() - meta_step_timer))
            print(" ** step_time: %s seconds." % elapsed)


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
    plt.savefig("Accuracies_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
    plt.grid()

    #train_set, test_images, test_labels = dataset.get_mini_dataset(
    #    eval_batch_size, eval_iters, shots, num_classes, num_channels, split="training"
    #)

    # Display the training accuracies.
    plt.figure()
    x = np.arange(0, int(len(losses)), 1)
    plt.plot(x, losses)
    plt.plot(x, losses_mse)
    plt.legend(["train"])
    plt.savefig("Losses_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
    plt.grid()

    # Display the training accuracies.
    plt.figure()
    x = np.arange(0, len(losses_mse), 1)
    plt.plot(x, losses_mse)
    plt.legend(["train", "test"])
    plt.savefig("Losses_MSE_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
    plt.grid()

    for images, labels in train_set:
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
            losses.append(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
     
    #train_set, test_images, test_labels = blind_dataset.get_mini_dataset(
      #              eval_batch_size, eval_iters, shots, num_classes, num_channels, split="blind")
    test_preds = model.predict(test_images)
    tpr, fpr, auc, auc2, thres = ROCCurveCalculate(test_labels, test_images, model)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--') # k = color black
    plt.plot(fpr, tpr, label="AUC: %.3f" % auc, linewidth=3) # for color 'C'+str(j), for j[0 9]
    plt.legend(loc='lower right', ncol=1, mode="expand")
    plt.title('ROC for %s training samples' % (TR))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    
    plt.savefig("ROCLensDetectNet_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))
    test_preds = tf.argmax(test_preds).numpy()

    nrows = 1
    ncols = 5
    _, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))

    sample_keys = list(train_dataset.data.keys())

    #for i, ax in zip(range(nrows*ncols), axarr):
        #temp_image = test_images[i]
        #temp_image = toimage(temp_image)
        #temp_image = np.stack((test_images[i, :, :, 0],) * 3, axis=2)
        #temp_image *= 255
        #temp_image = np.clip(temp_image, 0, 255).astype("uint8")
        #ax.set_title(
        #    "Label : {}, Prediction : #{}".format(int(test_labels[i]), test_preds[i])
        #)
        #ax.imshow(temp_image)
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
    #plt.show()
    #plt.savefig("FINAL_PREDICTION_{}_samples_{}_shots_{}x{}_size_{}_meta_iters_{}_version_norm-{}.png". format(TR, shots, input_shape, input_shape, meta_iters, version, normalize))

except AssertionError as error:
    print(error)
#except:
    #print("\n\n\n ************ WARNING *********** \n\n ** Something went wrong during execution.\
     #        Please check it out later. ** Proceeding to move generated files...")
    #filemover(TR, version, shots, input_shape, meta_iters, normalize)
    pass

filemover(TR, version, shots, input_shape, meta_iters, normalize)

timee = int((time.perf_counter() - begin)/(60))
print('\n ** Mission accomplished in %s minutes.' % timee)
print("\n ** FINISHED! ************************")
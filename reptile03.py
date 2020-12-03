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
from data_generator03 import Dataset
from rept_utilities import data_downloader, ROCCurveCalculate, toimage, filemover, fileremover, FScoreCalc
from rept_utilities import save_image, save_clue, conv_window
from rept_utilities import examples_graph, roc_curve_graph, acc_graph, loss_graph, roc_curve_graph_series

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')
##################################################
#PARAMETERS SECTION
###################################################
learning_rate = 0.001 ###########ORIGINALLY 0.003 - CHANGED ON VERSION 4
meta_step_size = 0.25

inner_batch_size = 25    ####ORIGINALLY 25     -100
eval_batch_size = 25    ###ORIGINALLY 25   -100

meta_iters = 5000        #ORIGINALLY 2000    -5000
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
input_shape = 28  #originally 28 for omniglot
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
data_downloader() 
###SAVE_CSV WITH CODE DATA.
with open('Code_data_version_%s.csv' % version, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(code_data)

###IMPORT LENSING DATASET TO CODE
#path = os.getcwd() + "/" + "lensdata/"
#labels = pd.read_csv(path + 'y_data20000fits.csv',delimiter=',', header=None)
#y_data = np.array(labels, np.uint8)
#x_datasaved = h5py.File(path + 'x_data20000fits.h5', 'r')
#x_data = x_datasaved['data']

#index = save_clue(x_data, y_data, TR, version, 1, input_shape, 5, 5, index)
#x_data = (x_data - np.nanmin(x_data))/np.ptp(x_data)

print("\n ** Building dataset functions:")
print(" ** Train_dataset is bein imported...")
train_dataset = Dataset(split="train", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(train_dataset)
print(" ** Test_dataset is bein imported...")
test_dataset = Dataset(split="test", version=version, TR=TR, 
vallim=vallim, index=index, input_shape=input_shape, num_channels=num_channels)
print(test_dataset)


print(" ** Network building stage...")

def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return layers.ReLU()(x)
    #return keras.activations.softmax(x)
    #return keras.activations.sigmoid(x)

inputs = layers.Input(shape=(input_shape, input_shape, num_channels))
x = conv_bn(inputs)
x = conv_bn(x)
x = conv_bn(x)
x = conv_bn(x)
x = layers.Flatten()(x)
outputs = layers.Dense(num_classes, activation=
output_layer)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
##model.compile()  <-- ORIGINAL!
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(loss= 'categorical_crossentropy', optimizer=optimizer)
#VERSION 23: model.compile(loss= 'binary_crossentropy', optimizer=optimizer , metrics=[ 'accuracy' ])
#plot_model(model,  to_file="model_REPTILE_version_%s.png" % version)

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
                    with tf.GradientTape() as tape:
                        preds = model(images)
                        #print("preds: ", preds)
                        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                        #print("loss: ", loss)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                test_preds = model.predict(test_images)
                mean_loss.append(log_loss(test_labels, test_preds))
                test_preds = tf.argmax(test_preds).numpy()
                num_correct = (test_preds == test_labels).sum()
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / num_classes)
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
    lauc, AUCall, FPRall, TPRall, f1s, f001s = ([] for i in range(6))
    for j in range(num_classes):
        test_l = (test_labels == j)
        print(test_l)
        tpr, fpr, auc, auc2, thres = ROCCurveCalculate(test_l, test_images, model)
        lauc = np.append(lauc, auc)
        AUCall.append(auc2)
        FPRall.append(fpr)
        TPRall.append(tpr)
        roc_curve_graph_series(fpr, tpr, auc, TR, shots, input_shape, meta_iters, version, normalize, j)

        #f1_score, f001_score = FScoreCalc(test_l, test_images, model)
        #f1s.append(f1_score)
        #f001s.append(f001_score)

    #writer.writerows([["med_f1", np.percentile(f1s, 50.0)],
    #                 ["low_f1", np.percentile(f1s, 15.87)],
    #                 ["hig_f1", np.percentile(f1s, 84.13)],
    #                 ["med_f001", np.percentile(f001s, 50.0)],
     #                ["low_f001", np.percentile(f001s, 15.87)],
     #                ["hig_f001", np.percentile(f001s, 84.13)]])

    print('\n ** Generating ultimate ROC graph...')
    medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--') # k = color black

    mauc = np.percentile(lauc, 50.0)
    mAUCall = np.percentile(AUCall, 50.0)
    plt.title('Median ROC over %s characters' % (num_classes))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)

    for num in range(0,int(thres),1):
        lis = [item[num] for item in TPRall]
        los = [item[num] for item in FPRall]
            
        medians_x.append(np.percentile(los, 50.0))
        medians_y.append(np.percentile(lis, 50.0))
        lowlim.append(np.percentile(lis, 15.87))
        highlim.append(np.percentile(lis, 84.13))
        
    lowauc = metrics.auc(medians_x, lowlim)
    highauc = metrics.auc(medians_x, highlim)

    print(lowauc, mAUCall, highauc)

    plt.plot(medians_x, medians_y, 'b', label = 'AUC: %s' % mauc, linewidth=3)  
    plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha=0.3, interpolate=True)
    plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha=0.3, interpolate=True)
    plt.legend(loc='lower right', ncol=1, mode="expand")

    plt.savefig("ROCLensDetectNet_Full_%s.png" % TR)

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
    #filemover(TR, version, shots, input_shape, meta_iters, normalize, output_layer)
    pass

filemover(TR, version, shots, input_shape, meta_iters, normalize, activation_layer, output_layer, maxpooling)

timee = int((time.perf_counter() - begin)/(60))
print('\n ** Mission accomplished in %s minutes.' % timee)
print("\n ** FINISHED! ************************")
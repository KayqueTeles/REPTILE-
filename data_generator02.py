import matplotlib.pyplot as plt, numpy as np, random, tensorflow as tf, cv2, os, scipy.misc, h5py, pandas as pd
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import sklearn, bisect

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers import Input
from collections import Counter
from rept_utilities import TestSamplesBalancer, toimage, save_image, save_clue

class Dataset:
    def __init__(self, training, version, TR, vallim, index, input_shape, num_channels):
        split = "train" if training else "test"
        self.data = {}
          
        path = os.getcwd() + "/" + "lensdata/"
        labels = pd.read_csv(path + 'y_data20000fits.csv',delimiter=',', header=None)
        y_data = np.array(labels, np.uint8)
        x_datasaved = pd.read_hdf(path + 'x_data20000fits.h5')

        print("\n ** Let's analyze this data!")
        x_datasaved.describe()
        x_datasaved.header()
        x_datasaved = h5py.File(path + 'x_data20000fits.h5', 'r')
        Ni_channels = 0 #first channel
        N_channels = 3 #number of channels

        x_data = x_datasaved['data']
        x_data = x_data[:,:,:,Ni_channels:Ni_channels + N_channels]

        x_data = (x_data - np.nanmin(x_data))/np.ptp(x_data)
        y_data, x_data = TestSamplesBalancer(y_data, x_data, vallim, TR, split)
        save_clue(x_data, y_data, TR, version, input_shape, 5, 5)

        for y in range(int(len(y_data))):
            image = x_data[y,:,:,:]
            print(image.shape)
            #last addicione
            #image = toimage(image)
            image = np.array(image)
            print(image.shape)
            #image = tf.image.convert_image_dtype(image, tf.float32)
            image = (image - np.nanmin(image))/np.ptp(image)
            label = str(y_data[y])
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
            self.labels = list(self.data.keys())
        ###

        print(" ** I HAVE A DATA VECTOR! ")
        print(" ** I HAVE A SHAPE VECTOR! ")
        ############################################3

    def get_mini_dataset(
        self, batch_size, repetitions, shots, num_classes, num_channels, split=False):
        #print(" ** Now we're using get_mini_dataset...")
        #print(split)
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 101, 101, num_channels))#, num_channels))
        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 101, 101, num_channels))
        #print(temp_images.shape)

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)
        #print(label_subset)
        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            #print(temp_labels)
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                test_images[class_idx] = images_to_split[-1]
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = images_to_split[:-1]
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)
            #print(temp_images)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        #print(dataset)
        if split:
            return dataset, test_images, test_labels
        return dataset
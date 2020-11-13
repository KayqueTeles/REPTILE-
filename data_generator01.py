import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds
import os, scipy.misc
import h5py
import pandas as pd
from rept_utilities import TestSamplesBalancer, imadjust, convert, toimage, save_image, save_clue
from PIL import Image
import cv2

class Dataset:
    def __init__(self, training, version, TR, vallim, index, input_shape):
        split = "train" if training else "test"
        self.data = {}
          
        path = os.getcwd() + "/" + "lensdata/"
        labels = pd.read_csv(path + 'y_data20000fits.csv',delimiter=',', header=None)
        y_data = np.array(labels, np.uint8)
        x_datasaved = h5py.File(path + 'x_data20000fits.h5', 'r')
        Ni_channels = 0 #first channel
        N_channels = 3 #number of channels

        x_data = x_datasaved['data']
        x_data = x_data[:,:,:,Ni_channels:Ni_channels + N_channels]

        y_data, x_data = TestSamplesBalancer(y_data, x_data, vallim, TR, split)
        x_data = (x_data - np.nanmin(x_data))/np.ptp(x_data)

        for y in range(int(len(y_data))):
            image = x_data[y,:,:,:]
            print(image.shape)
            image = np.array(image)
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
        self, batch_size, repetitions, shots, num_classes, split=False):
        #print(" ** Now we're using get_mini_dataset...")
        #print(split)
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 101, 101, 3))
        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 101, 101, 3))
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

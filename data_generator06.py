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
from tensorflow.keras.utils import to_categorical
from collections import Counter
from rept_utilities import toimage, save_image, save_clue, resize_image, extraction
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self, split, version, TR, vallim, index, input_shape, num_channels):
        ds = tfds.load("omniglot", split=split, as_supervised=True, shuffle_files=False)
        # Iterate over the dataset to get each individual image and its class,
        # and put that data into a dictionary.
        self.data = {}

        for image, label in ds.map(extraction):
            image = tf.image.resize(image, [input_shape, input_shape], preserve_aspect_ratio=True)
            image = np.array(image)
            print(image.shape)
            #image = image
            label = str(label)
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
            self.labels = list(self.data.keys())

    def get_mini_dataset(
        self, batch_size, repetitions, shots, num_classes, input_shape, num_channels, split="test"):
        print(" ** Now we're using get_mini_dataset...")
        print(split)
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, input_shape, input_shape, num_channels))#, num_channels))
        if split == "training":
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, input_shape, input_shape, num_channels))
            print(" test_images: ", test_images.shape)
            print(" test_labels: ", test_labels.shape)

        print(" temp_images: ", temp_images.shape)
        print(" temp_labels: ", temp_labels.shape)

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)
        print(" label_subset: %s" % label_subset)
        for class_idx, class_obj in enumerate(label_subset):
            print("class_idx: %s, class_obj: %s" % (class_idx, class_obj))
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            print(" temp_labels: %s" % temp_labels)
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if split == "training":
                test_labels[class_idx] = class_idx
                print(" test_labels: %s" % test_labels)
                images_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                print(" images_to_split: ", len(images_to_split))
                test_images[class_idx] = images_to_split[-1]
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = images_to_split[:-1]
                print(" temp_images: ", temp_images.shape)
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                temp_images[
                class_idx * shots : (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)
                print(" temp_images: ", temp_images.shape)
            #print(temp_images)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        #print(dataset)
        if split == "training":
            return dataset, test_images, test_labels
        return dataset

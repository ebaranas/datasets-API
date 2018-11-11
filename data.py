from mnist import MNIST
import os
import numpy as np

def read_raw_mnist(data_path):
    mndata = MNIST(data_path)
    images, labels = mndata.load_training()
    return np.asarray(images), np.asarray(labels)

def read_raw_bfimage(data_path):
    images = []
    labels = []
    for label_index, label in sorted(enumerate(os.listdir(data_path))):
        filename = data_path + label
        label = label.replace(".npy", "")
        for image in np.load(filename): # will load one npy file which may contain many examples
            images.append(image)
            labels.append(label_index)
    return np.asarray(images), np.asarray(labels)

SUPPORTED_DATA = {"mnist": read_raw_mnist, "bfimage": read_raw_bfimage}

def read_raw(data_name, data_path):
    read = (SUPPORTED_DATA.get(data_name, 'KeyError'))
    return read(data_path)
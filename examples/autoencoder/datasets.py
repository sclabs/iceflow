import numpy as np
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data


def mnist_ae():
    # load mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # make Datasets
    train_dataset = Dataset.from_tensor_slices(
        ({'x': mnist.train._images[:320, :]}, mnist.train._images[:320, :]))
    test_dataset = Dataset.from_tensor_slices(
        ({'x': mnist.test._images[:320, :]}, mnist.test._images[:320, :]))

    return train_dataset, test_dataset


def random_image_ae():
    image = np.random.random((784,)).astype(np.float32)
    return None, Dataset.from_tensors({'x': image})


def random_embedding():
    embedding = np.random.random((10,)).astype(np.float32)
    return None, Dataset.from_tensors({'z': embedding})


def random():
    image = np.random.random((784,)).astype(np.float32)
    embedding = np.random.random((10,)).astype(np.float32)
    return None, Dataset.from_tensors({'x': image, 'z': embedding})

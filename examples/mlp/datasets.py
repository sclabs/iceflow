import numpy as np
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data


def mnist():
    # load mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # make Datasets
    train_dataset = Dataset.from_tensor_slices(
        (mnist.train._images, mnist.train._labels))
    test_dataset = Dataset.from_tensor_slices(
        (mnist.test._images, mnist.test._labels))

    return train_dataset, test_dataset


def random_image():
    return None, Dataset.from_tensors(
        np.random.random((784,)).astype(np.float32))


def random_images():
    return None, Dataset.from_tensor_slices(
        np.random.random((32, 784,)).astype(np.float32))

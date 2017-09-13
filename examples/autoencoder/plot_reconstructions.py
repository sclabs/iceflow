import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import iceflow

from datasets import mnist_ae


def main():
    # make Estimator and Datasets
    e = iceflow.make_estimator('test.cfg')
    train, test = mnist_ae()

    # prepare input_fn
    input_fn = iceflow.make_input_fn(test, take=32)

    # run inputs
    with tf.Session() as sess:
        inputs, _ = sess.run(input_fn())

    # run inference for outputs
    outputs = np.stack(list(e.predict(input_fn)))

    # tile and imsave
    tiled_inputs = np.concatenate(inputs['x'].reshape((32, 28, 28)), axis=1)
    tiled_outputs = np.concatenate(outputs.reshape((32, 28, 28)), axis=1)
    plt.imsave('test.png', np.concatenate((tiled_inputs, tiled_outputs)),
               cmap='Greys', vmin=0, vmax=1)


if __name__ == '__main__':
    main()

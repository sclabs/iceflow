import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import iceflow

from datasets import mnist_ae


def main():
    # make Estimator and Datasets
    e = iceflow.make_estimator('test.cfg')
    train, test = mnist_ae()

    # get the first minibatch of test inputs
    with tf.Session() as sess:
        inputs, _ = sess.run(test.batch(32).make_one_shot_iterator().get_next())

    # run inference
    outputs = np.stack(list(e.predict(
        tf.estimator.inputs.numpy_input_fn(inputs, shuffle=False))))

    # tile and imsave
    tiled_inputs = np.concatenate(inputs['x'].reshape((32, 28, 28)), axis=1)
    tiled_outputs = np.concatenate(outputs.reshape((32, 28, 28)), axis=1)
    plt.imsave('test.png', np.concatenate((tiled_inputs, tiled_outputs)),
               cmap='Greys', vmin=0, vmax=1)


if __name__ == '__main__':
    main()

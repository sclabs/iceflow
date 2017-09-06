IceFlow
=======

_ice floe, nowhere to go_

A lightweight meta-framework for training neural networks with [TensorFlow](https://www.tensorflow.org/).

Installation
------------

    pip install iceflow

### Dependencies

 - `tensorflow>=1.3.0`
 - `dm-sonnet>=1.11`

Quick start
-----------

1. Define [Sonnet modules](https://deepmind.github.io/sonnet/) in `models.py`:

       import tensorflow as tf
       import sonnet as snt
       
       
       class MLP(snt.AbstractModule):
           def __init__(self, hidden_size, output_size, nonlinearity=tf.tanh):
               super(MLP, self).__init__()
               self._hidden_size = hidden_size
               self._output_size = output_size
               self._nonlinearity = nonlinearity
       
           def _build(self, inputs):
               lin_x_to_h = snt.Linear(output_size=self._hidden_size, name="x_to_h")
               lin_h_to_o = snt.Linear(output_size=self._output_size, name="h_to_o")
               return lin_h_to_o(self._nonlinearity(lin_x_to_h(inputs)))


2. Define [Datasets](https://www.tensorflow.org/programmers_guide/datasets)
   in `datasets.py`:

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

3. Describe what you want to do in `test.cfg`:

       [DEFAULT]
       model_dir=test1
       model=MLP
       loss=softmax_cross_entropy
       metrics=accuracy,auc
       hidden_size=50
       output_size=10

4. Train your model, evaluating every 1000 steps:

       $ iceflow train test.cfg mnist --eval_period 1000

5. Evaluate your model: #TODO: update this to include AUC metric

       $ iceflow eval test.cfg mnist
       {'global_step': 10000, 'loss': 0.13652229, 'accuracy': 0.96079999}

6. Visualize your learning in TensorBoard:

       $ tensorboard --logdir=test1

   Navigate to <http://localhost:6006> to see the metrics:

   ![](images/tensorboard.png)

7. Add some new data to `datasets.py`

       import numpy as np


       def random_image():
           return None, Dataset.from_tensors(
               np.random.random((784,)).astype(np.float32))
       
       
       def random_images():
           return None, Dataset.from_tensor_slices(
               np.random.random((32, 784,)).astype(np.float32))

   and make predictions on it
   
       $ iceflow predict test1.cfg random_image
       [5]
       
       $ iceflow predict test1.cfg random_images
       [5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 3, 5]

Config format reference
-----------------------

The format of the `iceflow` config file is roughly

    [DEFAULT]
    model_dir=test1
    model=MLP
    onehot=classes.txt
    loss=softmax_cross_entropy
    optimizer=AdagradOptimizer
    learning_rate=0.001
    optimizer_kwargs={initial_accumulator_value: 0.01}
    metrics=accuracy,auc
    hidden_size=50
    output_size=10

    [more_hiddens]
    model_dir=test2
    optimizer=AdagradOptimizer
    learning_rate=exponential_decay
    learning_rate_kwargs={
        learning_rate: 0.1,
        decay_steps: 1,
        decay_rate: 0.9}
    hidden_size=100

`model_dir` should be a unique folder name to write checkpoints to.

`model` must refer to a Sonnet module defined in `models.py`.

`onehot` can be skipped or set to False if your labels are not one-hot encoded.
If your labels are one-hot encoded, set this to True (to report classification
results as the one-hot indices of the classes) or set it to the name of a text
file on the disk whose `i`th line is the name of the class encoded at index `i`
(to report classification results as strings).

`loss` must be one of the losses defined in the [`tf.losses module`](https://www.tensorflow.org/api_docs/python/tf/losses).

`optimizer` must be one of the subclasses of [`tf.train.Optimizer`](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer)
defined in the [`tf.train` module](https://www.tensorflow.org/api_docs/python/tf/train).
If it is not passed it will default to [`tf.train.AdamOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer).

If `optimizer` requires a `learning_rate` parameter, you can either specify a
fixed learning rate (e.g., `learning_rate=0.001`) or one of the learning rate
decay schedulers in the [`tf.train` module](https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate)
(e.g., `learning_rate=exponential_decay`). These decay schedulers require extra
configuration, which should be specified in `learning_rate_kwargs`.

If you wish to pass additional kwargs to `optimizer`, you can do so in
`optimizer_kwargs`.

`metrics` can be skipped if you don't care about any evaluation metrics besides
the loss (which is always reported). If you do want to see additional metrics,
set this option to a comma-separated list of metrics defined in the
[`tf.metrics` module](https://www.tensorflow.org/api_docs/python/tf/metrics).

Every other key in the section is taken to be a hyperparameter which will be
passed as a kwarg to the constructor of the Sonnet module.

To train the model defined in the `[DEFAULT]` section, simply run

    $ iceflow train <config_file> <dataset>

The `[more_hiddens]` variant model inherits all hyperparameters from the
`[DEFAULT]` section but overrides some of them, including `model_dir` (to avoid
conflicting with the `[DEFAULT]` model). To train that one, run

    $ iceflow train <config_file> <dataset> --config_section more_hiddens

Design philosophy
-----------------

Our typical workload involves training lots of models (usually with complex or
experimental architecture) with different sets of hyperparameters on different
datasets.

Previously, we had been using a hand-built meta-framework around TensorFlow to
organize training, evaluation, and inference.

As of TensorFlow 1.3, the [Dataset API](https://www.tensorflow.org/programmers_guide/datasets),
[Estimator API](https://www.tensorflow.org/programmers_guide/estimators), and
[DeepMind's Sonnet library](https://deepmind.github.io/sonnet/) have arisen as
mature alternatives to our hand-crafted solutions.

IceFlow aims to provide the small bit of code needed to get these three APIs to
work together seamlessly - without sacrificing flexibility - and provide an
efficient "command line and config file"-based interface to the basic train, 
eval, predict cycle.

Caveats and future directions
-----------------------------

 - Currently, the only possible output you can obtain from `iceflow predict` is
   tensors being printed to the command line. We plan to extend this to allow
   specification of an arbitrary Python function that takes the prediction
   results (arrays) as input.
 - Currently, there is no easy way to use IceFlow to inject a properly-restored
   Estimator into arbitrary Python code. We plan to add a Python API to IceFlow
   in the near future.
 - Currently, performing validation every so often during training is very
   awkward. We are awaiting the return of [ValidationMonitor](https://www.tensorflow.org/get_started/monitors#configuring_a_validationmonitor_for_streaming_evaluation)
   from its banishment in the desert of deprecation (and following
   [this GitHub issue](https://github.com/tensorflow/tensorflow/issues/7669)).

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

3. Describe what you want to do in `test1.cfg`:

       [DEFAULT]
       model_dir=test1
       model=MLP
       hidden_size=50
       output_size=10

4. Train your model, evaluating every 1000 steps:

       $ iceflow train test1.cfg mnist --eval_period 1000

5. Evaluate your model:

       $ iceflow eval test1.cfg mnist
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
    hyperparam_1=50
    hyperparam_2=10
    
    [more_hiddens]
    model_dir=test2
    hyperparam_1=100

To train the model defined in the `[DEFAULT]` section, run

    $ iceflow train <config_file> <dataset>

To train the `[more_hiddens]` variant model, which inherits all hyperparameters
from the `[DEFAULT]` section but overrides `model_dir` (to avoid conflicting
with the `[DEFAULT]` model) and `hyperparam_1`, run

    $ iceflow train <config_file> <dataset> --config_section more_hiddens

`model` must refer to a Sonnet module defined in `models.py`.

Every key besides `model_dir` and `model` is taken to be a hyperparameter which
will be passed as a kwarg to the constructor of the Sonnet module.

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

 - Currently, the only supported type of problem is a softmax classification
   problem with one-hot labels. We plan to extend this.
 - Currently, the only possible output you can obtain from `iceflow predict` is
   tensors being printed to the command line. We plan to extend this to allow
   specification of an arbitrary Python function that takes the prediction
   results (arrays) as input.
 - Currently, the optimizer used for training is hard-coded. We plan to expose
   this as a parameter either in the config or on the command line. We also plan
   to extend this to support learning rate decay and related use cases.
 - Currently, there is no easy way to use IceFlow to inject a properly-restored
   Estimator into arbitrary Python code. We plan to add this capability.
 - Currently, the batch size and shuffle buffer size are not exposed. We plan to
   expose this soon.
 - Currently, performing validation every so often during training is very
   awkward. We are awaiting the return of [`ValidationMonitor`](https://www.tensorflow.org/get_started/monitors#configuring_a_validationmonitor_for_streaming_evaluation)
   from its banishment in the desert of deprecation (and following
   [this GitHub issue](https://github.com/tensorflow/tensorflow/issues/7669)).

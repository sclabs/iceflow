import sys
import argparse
import configparser

import tensorflow as tf


def make_input_fn(dataset, num_epochs=None, batch_size=32, shuffle=10000):
    def input_fn():
        d = dataset
        if shuffle:
            d = d.shuffle(shuffle)
        return d\
            .batch(batch_size)\
            .repeat(num_epochs)\
            .make_one_shot_iterator() \
            .get_next()
    return input_fn


def make_model_fn(model):
    def model_fn(features, labels, mode, params):
        model_instance = model(**params)
        logits = model_instance(features)
        predictions = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions)
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(tf.argmax(labels, 1), predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions, loss, train_op, eval_metric_ops)
    return model_fn


def iceflow():
    # argparse
    parser = argparse.ArgumentParser(prog='iceflow')
    parser.add_argument(
        'action',
        type=str,
        choices=['train', 'eval', 'predict'],
        help='''Action to perform.''')
    parser.add_argument(
        'config',
        type=str,
        help='''Config file to read model configuration from.''')
    parser.add_argument(
        'dataset',
        type=str,
        help='''The name of a function defined in dataset.py which returns a
        tuple of Datasets (train, test).''')
    parser.add_argument(
        '-c', '--config_section',
        type=str,
        default='DEFAULT',
        help='''Specify a section of the config file to read non-default params
        from. Default is 'DEFAULT' to read the DEFAULT section.''')
    parser.add_argument(
        '-s', '--steps',
        type=int,
        default=10000,
        help='''How many steps to train for. Default is 10000.''')
    parser.add_argument(
        '-p', '--checkpoint_period',
        type=int,
        default=0,
        help='''How often to checkpoint during training (in units of steps).
        Default is 0 for only checkpointing only at the end.''')
    parser.add_argument(
        '-e', '--eval_period',
        type=int,
        default=0,
        help='''How often to eval during training (in units of steps). Overrides
        -p/--checkpoint_period. Default is 0 for no eval during training.''')
    parser.add_argument(
        '-l', '--loglevel',
        type=str,
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
        default='INFO',
        help='''The logging level for tensorflow. Default is 'INFO'.''')
    args = parser.parse_args()

    # configure logging
    tf.logging.set_verbosity(getattr(tf.logging, args.loglevel))

    # configure sys.path
    sys.path.append('')

    # configparser
    config = configparser.ConfigParser()
    config.read(args.config)
    section = config[args.config_section]
    model_dir = section['model_dir']
    model = getattr(__import__('models'), section['model'])
    params = {k: v for k, v in section.items()
              if k not in ['model_dir', 'model']}

    # Dataset
    train, test = getattr(__import__('datasets'), args.dataset)()

    # Estimator
    e = tf.estimator.Estimator(
        model_fn=make_model_fn(model), model_dir=model_dir, params=params)

    # perform action
    if args.action == 'train':
        if args.eval_period:
            for _ in range(args.steps // args.eval_period):
                e.train(make_input_fn(train), steps=args.eval_period)
                print(e.evaluate(make_input_fn(test, num_epochs=1,
                                               shuffle=False)))
        elif args.checkpoint_period:
            for _ in range(args.steps // args.checkpoint_period):
                e.train(make_input_fn(train), steps=args.checkpoint_period)
        else:
            e.train(make_input_fn(train), steps=args.steps)
    elif args.action == 'eval':
        print(e.evaluate(make_input_fn(test, num_epochs=1, shuffle=False)))
    elif args.action == 'predict':
        print(list(e.predict(make_input_fn(test, num_epochs=1, shuffle=False))))

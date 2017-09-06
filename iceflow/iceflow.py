import sys
import argparse
import configparser
import json

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


def make_model_fn(model, loss_fn, optimizer_cls, optimizer_kwargs=None,
                  learning_rate=None, learning_rate_kwargs=None, onehot=False,
                  subgraph=None, metrics=()):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    def model_fn(features, labels, mode, params):
        model_instance = model(**params)
        if subgraph:
            logits = getattr(model_instance, subgraph)(features)
        else:
            logits = model_instance(features)
        if onehot:
            predictions = tf.argmax(logits, 1)
            if type(onehot) == str:
                table = tf.contrib.lookup.index_to_string_table_from_file(
                    onehot)
                predictions = table.lookup(predictions)
        else:
            predictions = logits
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions)
        loss = loss_fn(labels, logits)
        if hasattr(learning_rate, '__call__'):
            learning_rate_kwargs['global_step'] = tf.train.get_global_step()
            learning_rate_instance = learning_rate(**learning_rate_kwargs)
        else:
            learning_rate_instance = learning_rate
        optimizer_kwargs['learning_rate'] = learning_rate_instance
        optimizer = optimizer_cls(**optimizer_kwargs)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        eval_metric_ops = {}
        if onehot:
            labels = tf.argmax(labels, 1)
            logits = tf.argmax(logits, 1)
        for metric in metrics:
            eval_metric_ops[metric] = getattr(tf.metrics, metric)(labels,
                                                                  logits)
        return tf.estimator.EstimatorSpec(
            mode, predictions, loss, train_op, eval_metric_ops)
    return model_fn


def iceflow():
    # argparse
    parser = argparse.ArgumentParser(prog='iceflow')
    parser.add_argument(
        'action',
        type=str,
        choices=['train', 'eval', 'predict', 'encode', 'decode'],
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
        '-b', '--batch_size',
        type=int,
        default=32,
        help='''Set the batch size. Default is 32.''')
    parser.add_argument(
        '--shuffle_size',
        type=int,
        default=10000,
        help='''The size of the buffer to use when shuffling Datasets. Default
        is 10000.''')
    parser.add_argument(
        '-l', '--loglevel',
        type=str,
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
        default='INFO',
        help='''The logging level for tensorflow. Default is 'INFO'.''')
    args = parser.parse_args()
    if args.action in ['train', 'eval', 'predict']:
        subgraph = None
    else:
        subgraph = args.action

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
    onehot = False if 'onehot' not in section \
                      or section['onehot'].lower() == 'false' else \
        True if section['onehot'].lower() == 'true' else section['onehot']
    loss_fn = getattr(tf.losses, section['loss'])
    optimizer_cls = getattr(tf.train, section['optimizer']) \
        if 'optimizer' in section else tf.train.AdamOptimizer
    optimizer_kwargs = json.loads(section['optimizer_kwargs']) \
        if 'optimizer_kwargs' in section else {}
    learning_rate = None
    if 'learning_rate' in section:
        try:
            learning_rate = float(section['learning_rate'])
        except ValueError:
            learning_rate = getattr(tf.train, section['learning_rate'])
    learning_rate_kwargs = json.loads(section['learning_rate_kwargs']) \
        if 'learning_rate_kwargs' in section else None
    metrics = [e.strip() for e in section['metrics'].split(',')]\
        if 'metrics' in section else []
    params = {k: v for k, v in section.items()
              if k not in ['model_dir', 'model', 'onehot', 'loss', 'optimizer',
                           'optimizer_kwargs', 'learning_rate',
                           'learning_rate_kwargs', 'metrics']}

    # Dataset
    train, test = getattr(__import__('datasets'), args.dataset)()

    # Estimator
    e = tf.estimator.Estimator(
        model_fn=make_model_fn(
            model, loss_fn, optimizer_cls, optimizer_kwargs=optimizer_kwargs,
            learning_rate=learning_rate,
            learning_rate_kwargs=learning_rate_kwargs, onehot=onehot,
            subgraph=subgraph, metrics=metrics),
        model_dir=model_dir, params=params)

    # prepare input_fn kwargs
    train_input_kwargs = {
        'dataset': train,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle_size
    }
    test_input_kwargs = dict(train_input_kwargs, **{
        'dataset': test,
        'shuffle': False,
        'num_epochs': 1
    })

    # perform action
    if args.action == 'train':
        if args.eval_period:
            for _ in range(args.steps // args.eval_period):
                e.train(make_input_fn(**train_input_kwargs),
                        steps=args.eval_period)
                print(e.evaluate(make_input_fn(**test_input_kwargs)))
        elif args.checkpoint_period:
            for _ in range(args.steps // args.checkpoint_period):
                e.train(make_input_fn(**train_input_kwargs),
                        steps=args.checkpoint_period)
        else:
            e.train(make_input_fn(**train_input_kwargs), steps=args.steps)
    elif args.action == 'eval':
        print(e.evaluate(make_input_fn(**test_input_kwargs)))
    elif args.action in ['predict', 'encode', 'decode']:
        print(list(e.predict(make_input_fn(**test_input_kwargs))))

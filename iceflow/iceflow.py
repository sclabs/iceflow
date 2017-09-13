import sys
import argparse

import tensorflow as tf

from iceflow import make_estimator, make_input_fn


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

    # Dataset
    sys.path.append('')
    train, test = getattr(__import__('datasets'), args.dataset)()

    # Estimator
    e = make_estimator(args.config, section=args.config_section,
                       subgraph=subgraph)

    # prepare input_fn kwargs
    train_input_kwargs = {
        'dataset': train,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle_size,
        'num_epochs': None
    }
    test_input_kwargs = {
        'dataset': test,
        'batch_size': args.batch_size,
    }

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

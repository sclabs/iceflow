import sys
import configparser
import json

import tensorflow as tf

from iceflow.model_fn import make_model_fn


def make_estimator(configfile, section='DEFAULT', subgraph=None):
    # configure sys.path
    sys.path.append('')

    # configparser
    config = configparser.ConfigParser()
    config.read(configfile)
    section = config[section]
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
    learning_rate = 0.001
    if 'learning_rate' in section:
        try:
            learning_rate = float(section['learning_rate'])
        except ValueError:
            learning_rate = getattr(tf.train, section['learning_rate'])
    learning_rate_kwargs = json.loads(section['learning_rate_kwargs']) \
        if 'learning_rate_kwargs' in section else None
    metrics = [e.strip() for e in section['metrics'].split(',')] \
        if 'metrics' in section else []
    params = {k: v for k, v in section.items()
              if k not in ['model_dir', 'model', 'onehot', 'loss', 'optimizer',
                           'optimizer_kwargs', 'learning_rate',
                           'learning_rate_kwargs', 'metrics']}

    # Estimator
    return tf.estimator.Estimator(
        model_fn=make_model_fn(
            model, loss_fn, optimizer_cls, optimizer_kwargs=optimizer_kwargs,
            learning_rate=learning_rate,
            learning_rate_kwargs=learning_rate_kwargs, onehot=onehot,
            subgraph=subgraph, metrics=metrics),
        model_dir=model_dir, params=params)

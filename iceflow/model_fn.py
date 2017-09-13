import tensorflow as tf


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

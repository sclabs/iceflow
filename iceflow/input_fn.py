def make_input_fn(dataset, num_epochs=1, batch_size=32, shuffle=False,
                  take=None):
    def input_fn():
        d = dataset
        if shuffle:
            d = d.shuffle(shuffle)
        if take:
            d = d.take(take)
        return d\
            .batch(batch_size)\
            .repeat(num_epochs)\
            .make_one_shot_iterator() \
            .get_next()
    return input_fn

import tensorflow as tf
import sonnet as snt


class AE(snt.AbstractModule):
    def __init__(self, n_latent, n_out):
        super(AE, self).__init__()
        self._n_latent = n_latent
        self._n_out = n_out

    @snt.reuse_variables
    def encode(self, inputs):
        """Builds the front half of AutoEncoder, x -> z."""
        if type(inputs) == dict:
            inputs = inputs['x']
        w_enc = tf.get_variable("w_enc", shape=[self._n_out, self._n_latent])
        b_enc = tf.get_variable("b_enc", shape=[self._n_latent])
        return tf.sigmoid(tf.matmul(inputs, w_enc) + b_enc)

    @snt.reuse_variables
    def decode(self, inputs):
        """Builds the back half of AutoEncoder, z -> y."""
        if type(inputs) == dict:
            inputs = inputs['z']
        w_rec = tf.get_variable("w_dec", shape=[self._n_latent, self._n_out])
        b_rec = tf.get_variable("b_dec", shape=[self._n_out])
        return tf.sigmoid(tf.matmul(inputs, w_rec) + b_rec)

    def _build(self, inputs):
        """Builds the 'full' AutoEncoder, ie x -> z -> y."""
        latents = self.encode(inputs)
        return self.decode(latents)

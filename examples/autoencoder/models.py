import tensorflow as tf
import sonnet as snt


class AE(snt.AbstractModule):
    def __init__(self, n_hidden, n_latent, n_out, nonlinearity=tf.tanh):
        super(AE, self).__init__()
        self._n_hidden = n_hidden
        self._n_latent = n_latent
        self._n_out = n_out
        self._nonlinearity = nonlinearity

    @snt.reuse_variables
    def encode(self, inputs):
        """Builds the front half of AutoEncoder, x -> z."""
        if type(inputs) == dict:
            inputs = inputs['x']
        lin_x_to_h = snt.Linear(output_size=self._n_hidden, name="x_to_h")
        lin_h_to_z = snt.Linear(output_size=self._n_latent, name="h_to_z")
        return tf.sigmoid(lin_h_to_z(self._nonlinearity(lin_x_to_h(inputs))))

    @snt.reuse_variables
    def decode(self, inputs):
        """Builds the back half of AutoEncoder, z -> y."""
        if type(inputs) == dict:
            inputs = inputs['z']
        lin_z_to_h = snt.Linear(output_size=self._n_hidden, name="z_to_h")
        lin_h_to_y = snt.Linear(output_size=self._n_out, name="h_to_y")
        return tf.sigmoid(lin_h_to_y(self._nonlinearity(lin_z_to_h(inputs))))

    def _build(self, inputs):
        """Builds the 'full' AutoEncoder, ie x -> z -> y."""
        latents = self.encode(inputs)
        return self.decode(latents)

import tensorflow as tf
from keras import layers


class AddPositionEmbedding(layers.Layer):
    """
    Add position embedding to input. out = x + embedding
    """

    def __init__(self):
        super(AddPositionEmbedding, self).__init__()

    def build(self, input_shape):
        self.token_length = input_shape[-2]
        self.emb_dim = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.token_length, self.emb_dim],
            initializer='glorot_uniform',
            dtype=self.dtype,
            trainable=True)

    def call(self, input):
        v = tf.expand_dims(self.kernel, axis=0)
        return v + input

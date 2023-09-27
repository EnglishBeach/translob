"""
Realisation transformer from article
"""
import numpy as np
import tensorflow as tf
import keras

from typing import Union as _Union
from typing import Callable as _Callable
from keras.utils import get_custom_objects as _get_custom_objects
from keras import backend as _K



# DataClass = utilites.DataClass
# Input
def input_block(seq_len):
    inputs = keras.Input(shape=(seq_len, 40))
    return inputs


# CN
def cnn_block(
    input_layer,
    filters,
    dilation_steps,
):
    dilation_steps = [
        2**dilation
        for dilation in range(dilation_steps + 1)
    ] # yapf: disable
    x = input_layer
    for dilation in dilation_steps:
        layer = keras.layers.Conv1D(
            filters=filters,
            kernel_size=2,
            dilation_rate=dilation,
            activation='relu',
            padding='causal',
        )
        x = layer(x)
    return x


# Normalisation
def norm_block(input_layer):

    norm = keras.layers.LayerNormalization()(input_layer)
    return norm


# Positional encoding
class PositionalEncoding(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        steps, d_model = x.get_shape()[-2:]
        ps = np.zeros([steps, 1], dtype=_K.floatx())
        for step in range(steps):
            ps[step, :] = [(2 / (steps - 1)) * step - 1]

        ps_expand = _K.expand_dims(_K.constant(ps), axis=0)
        ps_tiled = _K.tile(ps_expand, [_K.shape(x)[0], 1, 1])

        x = _K.concatenate([x, ps_tiled], axis=-1)
        return x


def positional_encoder_block(input_layer):
    pos = PositionalEncoding()(input_layer)
    return pos


# Transformer
class MultiHeadSelfAttention(keras.layers.Layer):
    """
    Base class for Multi-head Self-Attention layers.
    """

    def __init__(self, num_heads: int, use_masking: bool, **kwargs):
        """
        :param num_heads: number of attention heads
        :param use_masking: when True, forbids the attention to see the further
          elements in the sequence.
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.qkv_weights = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['use_masking'] = self.use_masking
        return config

    def build(self, input_shape):
        # if not isinstance(input_shape, TensorShape):
        #     raise ValueError('Invalid input')
        d_model = input_shape[-1]

        self.validate_model_dimensionality(d_model)
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_uniform',
            trainable=True)

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        # if not K.is_keras_tensor(inputs):
        #     raise ValueError(
        #         'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = _K.int_shape(inputs)

        # Perform affine transformations to get the Queries, the Keys and the Values.
        qkv = _K.dot(inputs, self.qkv_weights)  # (-1,seq_len,d_model*3)
        qkv = _K.reshape(qkv, [-1, d_model * 3])

        # splitting the keys, the values and the queries.
        pre_q, pre_k, pre_v = [
            _K.reshape(
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)
        ]

        attention_out = self.attention(
            pre_q,
            pre_v,
            pre_k,
            seq_len,
            d_model,
            training=kwargs.get('training'),
        )
        # of shape (-1, seq_len, d_model)
        return attention_out

    def compute_output_shape(self, input_shape):
        shape_a, seq_len, d_model = input_shape
        return shape_a, seq_len, d_model

    def validate_model_dimensionality(self, d_model: int):
        if d_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')

    def attention(
        self,
        pre_q,
        pre_v,
        pre_k,
        seq_len: int,
        d_model: int,
        training=None,
    ):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        d_submodel = d_model // self.num_heads

        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = _K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = _K.permute_dimensions(pre_v, [0, 2, 1, 3])
        k = _K.permute_dimensions(pre_k, [0, 2, 3, 1])

        q = _K.reshape(q, (-1, seq_len, d_submodel))
        k = _K.reshape(k, (-1, seq_len, d_submodel))
        v = _K.reshape(v, (-1, seq_len, d_submodel))
        qk = tf.einsum('aib,ajb->aij', q, k)
        sqrt_d = _K.constant(np.sqrt(d_model // self.num_heads),
                             dtype=_K.floatx())
        a = qk / sqrt_d
        a = self.mask_attention(a)
        a = _K.softmax(a)
        attention_heads = tf.einsum('aij,ajb->aib', a, v)
        attention_heads = _K.reshape(attention_heads,
                                     (-1, self.num_heads, seq_len, d_submodel))
        attention_heads = _K.permute_dimensions(attention_heads, [0, 2, 1, 3])
        attention_heads = _K.reshape(attention_heads, (-1, seq_len, d_model))

        return attention_heads

    def mask_attention(self, dot_product):
        """
        Makes sure that (when enabled) each position
        (of a decoder's self-attention) cannot attend to subsequent positions.
        :param dot_product: scaled dot-product of Q and K after reshaping them
        to 3D tensors (batch * num_heads, rows, cols)
        """
        if not self.use_masking:
            return dot_product
        last_dims = _K.int_shape(dot_product)[-2:]
        low_triangle_ones = (
            np.tril(np.ones(last_dims))
            # to ensure proper broadcasting
            .reshape((1, ) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (
            _K.constant(low_triangle_ones, dtype=_K.floatx()) * dot_product +
            _K.constant(close_to_negative_inf * inverse_low_triangle))
        return result


_get_custom_objects().update({
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
})


class CustomNormalization(keras.layers.Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    """

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim, ),
            initializer='ones',
            trainable=True,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(dim, ),
            initializer='zeros',
            trainable=True,
        )
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = _K.mean(
            inputs,
            axis=self.axis,
            keepdims=True,
        )
        variance = _K.mean(
            _K.square(inputs - mean),
            axis=self.axis,
            keepdims=True,
        )
        epsilon = _K.constant(
            1e-5,
            dtype=_K.floatx(),
        )
        normalized_inputs = (inputs - mean) / _K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class TransformerTransition(keras.layers.Layer):
    """
    Transformer transition function. The same function is used both
    in classical in Universal Transformers.
    """

    def __init__(
        self,
        activation: _Union[str, _Callable],
        size_multiplier: int = 4,
        **kwargs,
    ):
        """
        :param activation: activation function. Must be a string or a callable.
        :param size_multiplier: How big the hidden dimension should be.
          Most of the implementation use transition functions having 4 times
          more hidden units than the model itself.
        :param kwargs: Keras-specific layer arguments.
        """
        self.activation = keras.activations.get(activation)
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['activation'] = keras.activations.serialize(self.activation)
        config['size_multiplier'] = self.size_multiplier
        return config

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.weights1 = self.add_weight(
            name='weights1',
            shape=(d_model, self.size_multiplier * d_model),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.biases1 = self.add_weight(
            name='biases1',
            shape=(self.size_multiplier * d_model),
            initializer='zeros',
            trainable=True,
        )
        self.weights2 = self.add_weight(
            name='weights2',
            shape=(self.size_multiplier * d_model, d_model),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.biases2 = self.add_weight(
            name='biases2',
            shape=(d_model, ),
            initializer='zeros',
            trainable=True,
        )
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = _K.int_shape(inputs)
        d_model = input_shape[-1]

        K_dot = _K.dot(_K.reshape(inputs, (-1, d_model)), self.weights1)
        step1 = self.activation(
            _K.bias_add(K_dot, self.biases1, data_format='channels_last'))

        K_dot = _K.dot(step1, self.weights2)
        step2 = _K.bias_add(K_dot, self.biases2, data_format='channels_last')
        result = _K.reshape(step2, (-1, ) + input_shape[-2:])
        return result


class TransformerLayer(keras.layers.Layer):
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each such block is, essentially:
    - Multi-head self-attention (masked or unmasked)
    - Residual connection,
    - Layer normalization
    - Transition function
    - Residual connection
    - Layer normalization
    """

    def __init__(
        self,
        # name: str,
        num_heads: int,
        use_masking: bool = True,
        **kwargs,
    ):
        self.attention_layer = MultiHeadSelfAttention(
            num_heads,
            use_masking=use_masking,
            # name=f'{name}_self_attention',
        )
        self.norm1_layer = CustomNormalization()
        self.norm2_layer = CustomNormalization()
        self.transition_layer = TransformerTransition(activation='relu', )
        self.addition_layer = keras.layers.Add()
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        #PostLN: X -> attention -> +X -> norm1 -> transition -> +norm1 -> norm2
        attention = self.attention_layer(x)
        residual_1 = (self.addition_layer([x, attention]))
        norm_1 = self.norm1_layer(residual_1)

        transition = self.transition_layer(norm_1)
        residual_2 = (self.addition_layer([norm_1, transition]))
        norm_2 = self.norm2_layer(residual_2)

        return norm_2


def transformer_block(
    input_layer,
    share_weights,
    n_blocks,
    n_heads,
):
    x = input_layer
    tb = TransformerLayer(
        num_heads=n_heads,
        use_masking=True,
    )
    for block in range(n_blocks):
        if share_weights:
            x = tb(x)
        else:
            x = TransformerLayer(
                num_heads=n_heads,
                use_masking=True,
            )(x)

    return x


# FFN
def ffn_block(
    input_layer,
    dropout_rate,
    activation,
    units,
    kernel_regularizer,
    kernel_initializer,
):
    input_layer = keras.layers.Flatten()(input_layer)

    input_layer = keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(input_layer)

    input_layer = keras.layers.Dropout(dropout_rate)(input_layer)
    out = keras.layers.Dense(
        units=3,
        activation='softmax',
    )(input_layer)
    return out


# Collection
class blocks:
    input_block = input_block
    cnn_block = cnn_block
    norm_block = norm_block
    positional_encoder_block = positional_encoder_block
    transformer_block = transformer_block
    ffn_block = ffn_block


# parametrs
PARAMETRS = {
    'seq_len': 100,
    'cn': dict(
        n_filters=14,
        dilation_steps=4,
    ),
    'an': dict(
        attention_heads=3,
        blocks=2,
        share_weights=False,
    ),
    'ff': dict(
        units = 64,
        dropout_rate=0.1,
        activation=keras.activations.relu,
        kernel_regularizer=keras.regularizers.L2(),
        kernel_initializer='glorot_uniform',
    ),
    'optimizer':
    keras.optimizers.legacy.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
    ),
} #yapf:disable


# build
def build_model(
    seq_len,
    cn__n_filters,
    cn__dilation_steps,
    an__blocks,
    an__attention_heads,
    an__share_weights,
    ff__units,
    ff__dropout_rate,
    ff__activation,
    ff__kernel_regularizer,
    ff__kernel_initializer,
    optimizer,
):
    # Model
    inputs = blocks.input_block(seq_len)
    x = inputs
    x = blocks.cnn_block(
        input_layer=x,
        filters=cn__n_filters,
        dilation_steps=cn__dilation_steps,
    )
    x = blocks.norm_block(input_layer=x)
    x = blocks.positional_encoder_block(input_layer=x)
    x = blocks.transformer_block(
        input_layer=x,
        n_blocks=an__blocks,
        n_heads=an__attention_heads,
        share_weights=an__share_weights,
    )
    x = blocks.ffn_block(
        input_layer=x,
        units=ff__units,
        dropout_rate=ff__dropout_rate,
        activation=ff__activation,
        kernel_regularizer=ff__kernel_regularizer,
        kernel_initializer=ff__kernel_initializer,
    )

    model = keras.Model(inputs=inputs, outputs=x)

    # Compile
    model.compile(
        optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='acc'),
            keras.metrics.SparseCategoricalAccuracy(name='sp_acc'),
        ],
    )
    return model

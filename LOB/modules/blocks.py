import numpy as _np
from typing import Union as _Union
from typing import Callable as _Callable

import keras as _keras
import tensorflow as _tf
from keras.utils import get_custom_objects as _get_custom_objects
from keras import backend as _K


# Input
def input_block(seq_len):
    inputs = _keras.Input(shape=(seq_len, 40))
    return inputs


# CN
def cnn_block(input_layer, filters=1, dilation_steps=0):
    dilation_steps = [
        2**dilation
        for dilation in range(dilation_steps + 1)
    ] # yapf: disable
    for dilation in dilation_steps:
        input_layer = _keras.layers.Conv1D(
            filters=filters,
            kernel_size=2,
            dilation_rate=dilation,
            activation='relu',
            padding='causal',
        )(input_layer)
        return input_layer


# Normalisation
def norm_block(input_layer):

    norm = _keras.layers.LayerNormalization()(input_layer)
    return norm


# Positional encoding
class _PositionalEncodingLayer(_keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        steps, d_model = x.get_shape()[-2:]
        ps = _np.zeros([steps, 1], dtype=_K.floatx())
        for step in range(steps):
            ps[step, :] = [(2 / (steps - 1)) * step - 1]

        ps_expand = _K.expand_dims(_K.constant(ps), axis=0)
        ps_tiled = _K.tile(ps_expand, [_K.shape(x)[0], 1, 1])

        x = _K.concatenate([x, ps_tiled], axis=-1)
        return x


def positional_encoder_block(input_layer):
    pos = _PositionalEncodingLayer()(input_layer)
    return pos


# Transformer
class _MultiHeadSelfAttention(_keras.layers.Layer):
    """
    Base class for Multi-head Self-Attention layers.
    """

    def __init__(self, num_heads: int, use_masking: bool,
                 **kwargs):
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
            for i in range(3)]

        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model,
                                       training=kwargs.get('training'))
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

    def attention(self, pre_q, pre_v, pre_k, seq_len: int, d_model: int, training=None):
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
        qk = _tf.einsum('aib,ajb->aij', q, k)
        sqrt_d = _K.constant(_np.sqrt(d_model // self.num_heads),
                            dtype=_K.floatx())
        a = qk / sqrt_d
        a = self.mask_attention(a)
        a = _K.softmax(a)
        attention_heads = _tf.einsum('aij,ajb->aib', a, v)
        attention_heads = _K.reshape(attention_heads, (-1, self.num_heads, seq_len, d_submodel))
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
            _np.tril(_np.ones(last_dims))
                # to ensure proper broadcasting
                .reshape((1,) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (
                _K.constant(low_triangle_ones, dtype=_K.floatx()) * dot_product +
                _K.constant(close_to_negative_inf * inverse_low_triangle))
        return result


_get_custom_objects().update(
    {'MultiHeadSelfAttention': _MultiHeadSelfAttention,})


class _LayerNormalization(_keras.layers.Layer):
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


class _TransformerTransition(_keras.layers.Layer):
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
        self.activation = _keras.activations.get(activation)
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['activation'] = _keras.activations.serialize(self.activation)
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


class _TransformerBlock(_keras.layers.Layer):
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
        name: str,
        num_heads: int,
        use_masking: bool = True,
        **kwargs,
    ):
        self.attention_layer = _MultiHeadSelfAttention(
            num_heads,
            use_masking=use_masking,
            name=f'{name}_self_attention',
        )
        self.norm1_layer = _LayerNormalization(name=f'{name}_norm1')
        self.norm2_layer = _LayerNormalization(name=f'{name}_norm2')
        self.transition_layer = _TransformerTransition(
            name=f'{name}_transition',
            activation='relu',
        )
        self.addition_layer = _keras.layers.Add(name=f'{name}_add')
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        output = self.attention_layer(x)
        post_residual1 = (self.addition_layer([x, output]))
        norm1_output = self.norm1_layer(post_residual1)
        output = self.transition_layer(norm1_output)
        post_residual2 = (self.addition_layer([norm1_output, output]))
        output = self.norm2_layer(post_residual2)
        return output


def transformer_block(input_layer,share_weights, n_blocks, n_heads):
    x=input_layer
    tb=_TransformerBlock(
            f'transformer_block',
            n_heads,
            True,
        )
    for block in range(n_blocks):
        if share_weights:
            x=tb(x)
        else:
            x = _TransformerBlock(
                f'transformer_block_{block}',
                n_heads,
                True,
            )(x)

    return x


# FFN
def ffn_block(input_layer, dropout_rate):
    input_layer = _keras.layers.Flatten()(input_layer)
    input_layer = _keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer='l2',
        kernel_initializer='glorot_uniform')(input_layer)
    input_layer = _keras.layers.Dropout(dropout_rate)(input_layer)
    out = _keras.layers.Dense(
        3,
        activation='softmax',
    )(input_layer)
    return out
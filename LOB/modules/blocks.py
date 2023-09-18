import numpy as _np

import keras as _keras
import tensorflow as _tf
from keras.utils import get_custom_objects

from modules.transformer import TransformerBlock as _TransformerBlock


# Input
def input_block(seq_len):
    inputs = _keras.Input(shape=(seq_len, 40))
    return inputs


# CN
def cnn_block(x, pars):
    dilation_steps = [
        2**dilation for dilation in range(pars.cn.dilation_steps + 1)
    ]
    for dilation in dilation_steps:
        x = _keras.layers.Conv1D(
            filters=pars.cn.n_filters,
            kernel_size=2,
            dilation_rate=dilation,
            activation='relu',
            padding='causal',
        )(x)
    # Normalisation
    norm = _keras.layers.LayerNormalization()(x)
    return norm


# Positional encoding
class _PositionalEncodingLayer(_keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        steps, d_model = x.get_shape()[-2:]
        global A
        A = x, steps, d_model
        ps = _np.zeros([steps, 1], dtype=_keras.backend.floatx())
        for step in range(steps):
            ps[step, :] = [(2 / (steps - 1)) * step - 1]

        ps_expand = _keras.backend.expand_dims(_keras.backend.constant(ps),
                                               axis=0)
        ps_tiled = _keras.backend.tile(ps_expand,
                                       [_keras.backend.shape(x)[0], 1, 1])

        x = _keras.backend.concatenate([x, ps_tiled], axis=-1)
        return x


def positional_encoder_block(x, pars):
    pos = _PositionalEncodingLayer()(x)
    return pos


# Transformers
def transformer(x, pars):
    for block in range(pars.an.blocks):
        x = _TransformerBlock(
            f'transformer_block_{block}',
            pars.an.attention_heads,
            True,
        )(x)
    return x


# FFN
def ffn_block(x, pars):
    x = _keras.layers.Flatten()(x)
    x = _keras.layers.Dense(64,
                            activation='relu',
                            kernel_regularizer='l2',
                            kernel_initializer='glorot_uniform')(x)
    x = _keras.layers.Dropout(pars.ff.dropout_rate)(x)
    out = _keras.layers.Dense(
        3,
        activation='softmax',
    )(x)
    return out
"""
Contains implementation of the Transformer model described in papers
"Attention is all you need" (https://arxiv.org/abs/1706.03762) and
"Universal Transformer" (https://arxiv.org/abs/1807.03819)
"""
# Based on Transformer from Keras-RL
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/transformer.py

from typing import Union, Callable
import numpy as np
import tensorflow as tf
import keras as _keras


class _MultiHeadSelfAttention(_keras.layers.Layer):
    """
    Base class for Multi-head Self-Attention layers.
    """

    def __init__(
        self,
        num_heads: int,
        use_masking: bool,
        **kwargs,
    ):
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
            trainable=True,
        )

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        # if not K.is_keras_tensor(inputs):
        #     raise ValueError(
        #         'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = _keras.backend.int_shape(inputs)

        # Perform affine transformations to get the Queries, the Keys and the Values.
        qkv = _keras.backend.dot(inputs,
                                 self.qkv_weights)  # (-1,seq_len,d_model*3)
        qkv = _keras.backend.reshape(qkv, [-1, d_model * 3])

        # splitting the keys, the values and the queries.
        pre_q, pre_k, pre_v = [
            _keras.backend.reshape(
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads),
            ) for i in range(3)
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
        q = _keras.backend.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = _keras.backend.permute_dimensions(pre_v, [0, 2, 1, 3])
        k = _keras.backend.permute_dimensions(pre_k, [0, 2, 3, 1])

        q = _keras.backend.reshape(q, (-1, seq_len, d_submodel))
        k = _keras.backend.reshape(k, (-1, seq_len, d_submodel))
        v = _keras.backend.reshape(v, (-1, seq_len, d_submodel))
        qk = tf.einsum('aib,ajb->aij', q, k)
        sqrt_d = _keras.backend.constant(np.sqrt(d_model // self.num_heads),
                                         dtype=_keras.backend.floatx())
        a = qk / sqrt_d
        a = self.mask_attention(a)
        a = _keras.backend.softmax(a)
        attention_heads = tf.einsum('aij,ajb->aib', a, v)
        attention_heads = _keras.backend.reshape(
            attention_heads,
            (-1, self.num_heads, seq_len, d_submodel),
        )
        attention_heads = _keras.backend.permute_dimensions(
            attention_heads, [0, 2, 1, 3])
        attention_heads = _keras.backend.reshape(attention_heads,
                                                 (-1, seq_len, d_model))

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
        last_dims = _keras.backend.int_shape(dot_product)[-2:]
        low_triangle_ones = (
            np.tril(np.ones(last_dims))
            # to ensure proper broadcasting
            .reshape((1, ) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (_keras.backend.constant(
            low_triangle_ones,
            dtype=_keras.backend.floatx(),
        ) * dot_product + _keras.backend.constant(
            close_to_negative_inf * inverse_low_triangle), )
        return result


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
        mean = _keras.backend.mean(
            inputs,
            axis=self.axis,
            keepdims=True,
        )
        variance = _keras.backend.mean(
            _keras.backend.square(inputs - mean),
            axis=self.axis,
            keepdims=True,
        )
        epsilon = _keras.backend.constant(
            1e-5,
            dtype=_keras.backend.floatx(),
        )
        normalized_inputs = (inputs - mean) / _keras.backend.sqrt(variance +
                                                                  epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class _TransformerTransition(_keras.layers.Layer):
    """
    Transformer transition function. The same function is used both
    in classical in Universal Transformers.
    """

    def __init__(self,
                 activation: Union[str, Callable],
                 size_multiplier: int = 4,
                 **kwargs):
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
        input_shape = _keras.backend.int_shape(inputs)
        d_model = input_shape[-1]
        step1 = self.activation(
            _keras.backend.bias_add(_keras.backend.dot(
                _keras.backend.reshape(inputs, (-1, d_model)), self.weights1),
                                    self.biases1,
                                    data_format='channels_last'), )
        step2 = _keras.backend.bias_add(_keras.backend.dot(
            step1, self.weights2),
                                        self.biases2,
                                        data_format='channels_last')
        result = _keras.backend.reshape(step2, (-1, ) + input_shape[-2:])
        return result


class TransformerBlock(_keras.layers.Layer):
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
            num_heads, use_masking=use_masking, name=f'{name}_self_attention')
        self.norm1_layer = _LayerNormalization(name=f'{name}_normalization1')
        self.norm2_layer = _LayerNormalization(name=f'{name}_normalization2')
        self.transition_layer = _TransformerTransition(
            name=f'{name}_transition', activation='relu')
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

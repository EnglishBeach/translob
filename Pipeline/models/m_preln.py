import keras
from . import m_base
from .m_base import blocks


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
        num_heads: int,
        use_masking: bool = True,
        **kwargs,
    ):
        self.attention_layer = m_base.MultiHeadSelfAttention(
            num_heads,
            use_masking=use_masking,
            # name=f'{name}_self_attention',
        )
        self.norm1_layer = m_base.CustomNormalization()
        self.norm2_layer = m_base.CustomNormalization()
        self.transition_layer = m_base.TransformerTransition(
            activation='relu', )
        self.addition_layer = keras.layers.Add()
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        #PreLN: X -> norm2 -> attention -> +X -> norm2 -> transition -> +(+X)

        norm_1 = self.norm1_layer(x)
        attention = self.attention_layer(norm_1)
        residual_1 = (self.addition_layer([x, attention]))

        norm_2 = self.norm2_layer(residual_1)
        transition = self.transition_layer(norm_2)
        residual_2 = (self.addition_layer([residual_1, transition]))

        return residual_2


def transformer_block(input_layer, share_weights, n_blocks, n_heads):
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


PARAMETRS = m_base.PARAMETRS


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
    x = transformer_block(
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
            keras.metrics.SparseCategoricalAccuracy(name='sp_acc'),
        ],
    )
    return model

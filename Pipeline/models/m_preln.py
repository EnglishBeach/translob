import tensorflow as tf
from . import m_base
from tensorflow.keras.utils import get_custom_objects as _get_custom_objects


class TransformerLayer(tf.keras.layers.Layer):
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
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.attention_layer = m_base.MultiHeadSelfAttention(
            num_heads,
            use_masking=use_masking,
            # name=f'{name}_self_attention',
        )
        self.norm1_layer = m_base.CustomNormalization()
        self.norm2_layer = m_base.CustomNormalization()
        self.transition_layer = m_base.TransformerTransition(
            activation='relu', )
        self.addition_layer = tf.keras.layers.Add()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "use_masking": self.use_masking,
        })
        return config
    def call(self, x, **kwargs):
        #PreLN: X -> norm2 -> attention -> +X -> norm2 -> transition -> +(+X)

        norm_1 = self.norm1_layer(x)
        attention = self.attention_layer(norm_1)
        residual_1 = (self.addition_layer([x, attention]))

        norm_2 = self.norm2_layer(residual_1)
        transition = self.transition_layer(norm_2)
        residual_2 = (self.addition_layer([residual_1, transition]))

        return residual_2


_get_custom_objects().update({
    'TransformerLayer': TransformerLayer,
})


def transformer_block(
    input_layer,
    *,
    share_weights,
    blocks,
    heads,
):
    x = input_layer
    tb = TransformerLayer(
        num_heads=heads,
        use_masking=True,
    )
    for block in range(blocks):
        if share_weights:
            x = tb(x)
        else:
            x = TransformerLayer(
                num_heads=heads,
                use_masking=True,
            )(x)

    return x


PARAMETRS = m_base.PARAMETRS


class blocks(m_base.blocks):

    def build_model(
        seq_len,
        convolutional,
        transformer,
        feed_forward,
        optimizer,
    ):
        # Model
        inputs = blocks.input_block(seq_len)
        x = inputs
        x = blocks.convolutional_block(x, **convolutional)
        x = blocks.norm_block(x)
        x = blocks.positional_encoder_block(x)
        x = transformer_block(x, **transformer)
        x = blocks.feed_forward_block(x, **feed_forward)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        # Compile
        model.compile(
            blocks.optimazer_block(**optimizer),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name='sp_acc'),
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
            ],
        )
        return model

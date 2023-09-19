import numpy as np
from tqdm import tqdm

import tensorflow as tf
import keras
from keras.preprocessing import timeseries_dataset_from_array
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report

from modules import blocks
from modules import data
from modules import utilites

# Model parametrs
class Input_pars(utilites.DataClass):
    seq_len = 100


class CN_pars(utilites.DataClass):
    n_filters = 14
    dilation_steps = 4  # dilation = 2**dilation_step


class AN_pars(utilites.DataClass):
    attention_heads = 3
    blocks = 2
    share_weights = True


class FF_pars(utilites.DataClass):
    dropout_rate = 0.1


class Optimaser_pars(utilites.DataClass):
    lr = 0.0001
    adam_beta1 = 0.9
    adam_beta2 = 0.999


class Trainig_pars(utilites.DataClass):
    shuffle = True
    batch_size = 512
    epochs = 150


class Full_pars(utilites.DataClass):
    seq_len = 100
    cn = CN_pars()
    an = AN_pars()
    ff = FF_pars()
    optimazer = Optimaser_pars()
    training = Trainig_pars()


pars = Full_pars()
# Dataset
print('Reading...')
((x_train, y_train), (x_val, y_val), (x_test, y_test)) =\
data.load_dataset(horizon=4)



ds_train = data.build_dataset(
    x=x_train,
    y=y_train,
    batch_size=pars.training.batch_size,
    seq_len=pars.seq_len,
)

ds_val = data.build_dataset(
    x=x_val,
    y=y_val,
    batch_size=pars.training.batch_size,
    seq_len=pars.seq_len,
)

ds_test = data.build_dataset(
    x=x_test,
    y=y_test,
    batch_size=pars.training.batch_size,
    seq_len=pars.seq_len,
)

# Model
print('Building...')
inputs = blocks.input_block(pars.seq_len)
cnn = blocks.cnn_block(
    input_layer=inputs,
    filters=pars.cn.n_filters,
    dilation_steps=pars.cn.dilation_steps,
)
norm = blocks.norm_block(input_layer=cnn)
pos = blocks.positional_encoder_block(input_layer=norm)

x = pos
transformer = blocks.transformer_block(
    input_layer=pos,
    n_blocks=pars.an.blocks,
    n_heads=pars.an.attention_heads,
    share_weights=True,
)
ffn = blocks.ffn_block(
    input_layer=transformer,
    dropout_rate=pars.ff.dropout_rate,
)
end = ffn



model = keras.Model(inputs=inputs, outputs=end)
model.summary()
print(
    f'Train x: {str(x_train.shape): <15} - y: {y_train.shape}',
    f'Val   x: {str(x_val.shape): <15} - y: {y_val.shape}',
    sep='\n',
)

# CompiLe
model.compile(
    keras.optimizers.Adam(
        learning_rate=pars.optimazer.lr,
        beta_1=pars.optimazer.adam_beta1,
        beta_2=pars.optimazer.adam_beta2,
        name="Adam",
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        keras.metrics.CategoricalAccuracy(name='accurancy'),
        keras.metrics.SparseCategoricalAccuracy(name='sparce_accurancy'),
    ],
)

# Train
model.fit(
    ds_train,
    epochs=pars.training.epochs,
    validation_data=ds_val,
    # callbacks=[
    #     CSVLogger('log.csv', append=True, separator=';')
    # ]
)

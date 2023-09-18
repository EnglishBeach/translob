import numpy as np
from tqdm import tqdm
import keras
from keras import Model
from keras.preprocessing import timeseries_dataset_from_array

from modules.utilites import DataClass
import modules.blocks as blocks
from modules.data_load import load_dataset

# from sklearn.metrics import accuracy_score, classification_report
# from keras.callbacks import ModelCheckpoint


# Model parametrs
class Input_pars(DataClass):
    seq_len = 100

class CN_pars(DataClass):
    n_filters=14
    dilation_steps=4  # dilation = 2**dilation_step

class AN_pars(DataClass):
    attention_heads = 3
    blocks = 2
    share_weights = True

class FF_pars(DataClass):
    dropout_rate = 0.1

class Optimaser_pars(DataClass):
    lr = 0.0001
    adam_beta1 = 0.9
    adam_beta2 = 0.999

class Trainig_pars(DataClass):
    batch_size = 512
    epochs = 150

class Full_pars(DataClass):
    cn= CN_pars()
    an = AN_pars()
    ff=FF_pars()
    optimazer=Optimaser_pars()
    training=Trainig_pars()

pars= Full_pars()

# Model
inputs = blocks._keras.Input()

model = Model(inputs=inputs, outputs=out)


# CompiLe
model.summary()

model.compile(
    keras.optimizers.Adam(
        learning_rate=pars.optimazer.lr,
        beta_1=pars.optimazer.adam_beta1,
        beta_2=pars.optimazer.adam_beta2,
        name="Adam",
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['sparse_categorical_accuracy'],
)
print(
    f'Train x: {str(x_train.shape): <15} - y: {y_train.shape}',
    f'Val   x: {str(x_val.shape): <15} - y: {y_val.shape}',
    sep='\n',
)

def set_shape(value_x,value_y):
    value_x.set_shape((None,100,40))
    # value_y.set_shape((None,))
    return value_x,value_y


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(horizon=4)

ds_train = timeseries_dataset_from_array(
    data=x_train,
    targets=y_train,
    batch_size=pars.training.batch_size,
    sequence_length=seq_len,
    shuffle=True,
)
ds_train= ds_train.map(set_shape)

ds_val = timeseries_dataset_from_array(
    data=x_val,
    targets=y_val,
    batch_size=pars.training.batch_size,
    sequence_length=seq_len,
    shuffle=True,
)
ds_val= ds_val.map(set_shape)


model.fit(
    ds_train,
    epochs=pars.training.epochs,
    validation_data=ds_val,
)

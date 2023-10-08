# %%
## For collab
# %tensorboard
# try:
#     from google.colab import drive
#     drive.mount('/content/drive/',force_remount=True)
#     %cd /content/drive/My Drive/LOB/
#     %pip install automodinit keras_tuner
#     !nohup /usr/bin/python3 /content/drive/MyDrive/LOB/Colab_saver.py &
# except: pass

# %%
import os
import datetime
import numpy as np
import tensorflow as tf
import keras_tuner

from tools import utils, express
from tools.utils import DataClass
from models import m_base as test_model

seq_len = 100

# %%
## Load data
proportion = input('Data proportion 100-0 in % (press enter for all): ')
if proportion == '': proportion = 1
else: proportion = float(proportion) / 100

row_data = express.load_saved_datas(proportion)
# row_data = data.load_datas(horizon,path=r'../dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore',)
express.inspect_datas(row_data)

datasets = express.build_datasets(
    datas=row_data,
    batch_size=100,
    seq_len=seq_len,
)
(ds_train, ds_val, ds_test) =\
(datasets['train'], datasets['val'], datasets['test'])
express.inspect_datasets(datasets)

# %%
DEFAULT_PARAMETRS= DataClass(test_model.PARAMETRS)
DEFAULT_PARAMETRS

# %%
## Tuner parametrs
# def configure(hp: keras_tuner.HyperParameters):

#     class CN_search(DataClass):
#         dilation_steps = hp.Int(
#             'dilation_steps',
#             default=4,
#             min_value=3,
#             max_value=5,
#             step=1,
#         )

#     class AN_search(DataClass):
#         share_weights = hp.Boolean(
#             'share_weights',
#             default=True,
#         )
#         blocks = hp.Int(
#             'an_blocks',
#             default=2,
#             min_value=1,
#             max_value=3,
#             step=1,
#         )

#     class Full_search(DataClass):
#         cn = CN_search()
#         an = AN_search()

#     return Full_search()


def configure_parametrs(hp: keras_tuner.HyperParameters):

    DEFAULT_PARAMETRS.convolutional.dilation_steps = 5

    DEFAULT_PARAMETRS.transformer.share_weights = hp.Boolean(
        'share_weights',
        default=True,
    )

    DEFAULT_PARAMETRS.feed_forward.activation = hp.Choice(
        name='activation',
        values=['relu', 'None'],
        default='relu',
    )
    DEFAULT_PARAMETRS.feed_forward.kernel_regularizer = hp.Choice(
        name='regularizer',
        values=['l2', 'None'],
        default='l2',
    )

    DEFAULT_PARAMETRS.optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=hp.Choice(name='lr',
                                default=0.0001,
                                values=[0.01, 0.001, 0.0005, 0.0001]),
        beta_1=0.9,
        beta_2=0.999,
    )

    return DEFAULT_PARAMETRS

# %%
## Build
def search_model(hp):
    parametrs = configure_parametrs(hp)
    model = test_model.blocks.build_model(**parametrs.Data_nested)
    return model


input_name = ''
date_tag = f'({datetime.datetime.now().strftime("%H-%M-%S--%d.%m")})'
while input_name == '':
    input_name = input(f"Input search name: ")
search_name = f'search_{input_name}{date_tag}'

print(
    f'Pattern model: {test_model.__name__}',
    f'Search name: {search_name}',
    'Parametrs:',
    DataClass(test_model.PARAMETRS),
    sep='\n',
)


# %%
##Callbacks
callback_freq = 100
model_dir = f'{express.callback_path}/{search_name}'
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=model_dir,
        histogram_freq=callback_freq,
        update_freq=callback_freq,
    ),
]

print(
    f"Callbacks:\n{[str(type(callback)).split('.')[-1] for callback in callbacks]}",
    f'Directory: {model_dir}',
    sep='\n',
)

# %%
## Build tuner
tuner = keras_tuner.GridSearch(
    hypermodel=search_model,
    objective="loss",
    executions_per_trial=1,
    directory=model_dir,
)

# %%
## Train
training_question = ''
while training_question not in ['y', 'n']:
    training_question = input('Start training now? (y-yes) (n-exit): ')
if training_question == 'y':
    tuner.search(
        ds_train,
        validation_data=ds_val,
        epochs=20,
        callbacks=callbacks,
    )

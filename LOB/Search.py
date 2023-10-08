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

from tools import data, utils
from tools.utils import DataClass
from models import m_base as test_model

seq_len = 100

# %%
## Load data
proportion = input('Data proportion 100-0 in % (press enter for all): ')
if proportion == '': proportion = 1
else: proportion = float(proportion) / 100

row_data = data.load_saved_datas(proportion)
# row_data = data.load_datas(horizon,path=r'../dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore',)
data.inspect_datas(row_data)

datasets = data.build_datasets(
    datas=row_data,
    batch_size=100,
    seq_len=seq_len,
)
(ds_train, ds_val, ds_test) =\
(datasets['train'], datasets['val'], datasets['test'])
data.inspect_datasets(datasets)

# %%
## Tuner parametrs
def configure(hp: keras_tuner.HyperParameters):

    class CN_search(DataClass):
        dilation_steps = hp.Int(
            'cn_dilation_steps',
            default=4,
            min_value=3,
            max_value=5,
            step=1,
        )

    class AN_search(DataClass):
        share_weights = hp.Boolean(
            'share_weights',
            default=True,
        )
        blocks = hp.Int(
            'an_blocks',
            default=2,
            min_value=1,
            max_value=3,
            step=1,
        )

    class Full_search(DataClass):
        cn = CN_search()
        an = AN_search()

    return Full_search()

# %%
## Build
from models import m_base as test_model


def search_model(hp):
    hyper_pars_data = configure(hp)
    pars_data = DataClass(test_model.PARAMETRS)
    pars = pars_data.Info_expanded
    pars.update(hyper_pars_data.Info_expanded)

    model = test_model.build_model(**pars)
    return model


input_name = ''
date_tag = f'({datetime.datetime.now().strftime("%H-%M-%S--%d.%m")})'
while input_name == '':
    input_name = input(f"Input search name: ")
search_name = f'search_{input_name}{date_tag}'

print(
    f'Pattern model name: {test_model.__name__}',
    f'Search name: {search_name}',
    'Deafult changed parametrs:',
    configure(keras_tuner.HyperParameters()),
    sep='\n',
)


# %%
##Callbacks
callback_freq = 100
model_dir = f'{data.callback_path}/{search_name}'
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=model_dir,
        histogram_freq=callback_freq,
        update_freq=callback_freq,
    ),
    # tf.keras.callbacks.ModelCheckpoint(
    #     f'{model_dir}/checkpoint/'+'{epoch:04d}.keras',
    #     monitor="val_sp_acc",
    #     verbose=0,
    #     save_best_only=False,
    #     save_weights_only=False,
    #     mode="auto",
    #     save_freq=callback_freq,
    # )
]

print(
    f"Callbacks:\n{[str(type(callback)).split('.')[-1] for callback in callbacks]}",
    f'Model directory: {model_dir}',
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

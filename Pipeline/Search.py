# %%
## For platforms
import os


def get_platform():
    platform = ''

    # Windows
    if os.name == 'nt':
        try:
            get_ipython().__class__.__name__
            platform = 'jupyter'
        except NameError:
            platform = 'python'

    elif os.name == 'posix':
        # Kaggle
        if 'KAGGLE_DATA_PROXY_TOKEN' in os.environ.keys():
            platform = 'kaggle'

    # Google Colab
        else:
            try:
                from google.colab import drive
                platform = 'colab'
            except ModuleNotFoundError:
                platform = None

    print(f'Use: {platform}')
    return platform


def colab_action():
    from google.colab import drive
    drive.mount('/content/drive/', force_remount=True)
    os.chdir(f'/content/drive/My Drive/LOB/Pipeline')
    os.system('pip install automodinit keras_tuner')
    os.system('nohup /usr/bin/python3 Colab_saver.py &')


def kaggle_action():
    ...


platform = get_platform()
if platform == 'colab':
    colab_action()
elif platform == 'kaggle':
    kaggle_action()


import backend as B
B.set_backend(platform)

# %%
# %tensorboard

# %%
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner

from backend import DataBack,ModelBack,DataClass

from models import m_base as test_model

seq_len = 100

# %%
## Savig data
# header = '../dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore'
# train_files= [
#         f'{header}_Training/Train_Dst_NoAuction_ZScore_CF_{i}.txt'
#         for i in range(7, 8)
#     ]

# test_files=[
#         f'{header}_Testing/Test_Dst_NoAuction_ZScore_CF_{i}.txt'
#         for i in range(1, 2)
#     ]

# all,test = Datasets.from_files(train_files)
# train, val = Datasets.validation_split(all)
# Datasets.inspect_data(train=train,val=val)
# Datasets.save_data(train=train,val=val)

# %%
## Load data
proportion = input('Data proportion 100-0 in % (press enter for all): ')
if proportion == '': proportion = 1
else: proportion = float(proportion) / 100

train, val, test = DataBack.from_saved(proportion=proportion,
                                       train_indexes=[0],
                                       val_indexes=[0])
DataBack.inspect_data(train=train, val=val, test=test)

ds_train = DataBack.build_dataset(data=train, seq_len=seq_len, batch_size=100)
ds_val = DataBack.build_dataset(data=val, seq_len=seq_len, batch_size=100)
DataBack.inspect_dataset(train=ds_train, val=ds_val)

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

    DEFAULT_PARAMETRS.transformer.share_weights = False

    choices = {'l2': 'l2', 'None': None}
    choice = hp.Choice(
        name='regularizer',
        values=list(choices),
        default='None',
    )
    DEFAULT_PARAMETRS.feed_forward.kernel_regularizer = choices[choice]

    lr = hp.Choice(
        name='lr',
        default=0.0001,
        values=[0.01, 0.001, 0.0005, 0.0001],
    )
    choices = {
        'sgd':
        tf.keras.optimizers.legacy.SGD(learning_rate=lr),
        'rms':
        tf.keras.optimizers.legacy.RMSprop(learning_rate=lr),
        'adam':
        tf.keras.optimizers.legacy.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
        ),
    }

    choice = hp.Choice(
        name='optimazer',
        default='adam',
        values=['adam', 'rms', 'sgd'],
    )
    DEFAULT_PARAMETRS.optimizer = choices[choice]
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
    configure_parametrs(keras_tuner.HyperParameters()),
    sep='\n',
)


# %%
##Callbacks
callback_freq = 100
model_dir = f"{ModelBack.callback_path}/{search_name}"
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

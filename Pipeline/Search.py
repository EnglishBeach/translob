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

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner

from backend import data_container, ModelBack, DataClass,dump_config_function

seq_len = 100

# %%
from models import m_preln as test_model

# %%
## Savig data
# header = '../dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore'
# train_files= [
#         f'{header}_Training/Train_Dst_NoAuction_ZScore_CF_{i}.txt'
#         for i in range(7, 8)
#     ]

# # test_files=[
# #         f'{header}_Testing/Test_Dst_NoAuction_ZScore_CF_{i}.txt'
# #         for i in range(1, 2)
# #     ]

# all,test = DataBack.from_files(train_files,horizon=4)
# train, val = DataBack.validation_split(all)
# DataBack.inspect_data(train=train,val=val)
# DataBack.save_data(train=train,val=val)

# %%
## Load data
proportion = input('Data proportion 100-0 in % (press enter for all): ')
if proportion == '': proportion = 1
else: proportion = float(proportion) / 100

train, val, test = data_container.read_saved_data(proportion=proportion,
                                       train_indexes=[0],
                                       val_indexes=[0])
data_container.inspect_data(train=train, val=val, test=test)

ds_train = data_container.data_to_dataset(data=train, seq_len=seq_len, batch_size=100)
ds_val = data_container.data_to_dataset(data=val, seq_len=seq_len, batch_size=100)
data_container.inspect_dataset(train=ds_train, val=ds_val)

# %%
PARAMETRS = DataClass(test_model.PARAMETRS)
PARAMETRS

# %%
## Tuner parametrs
def configure(hp: keras_tuner.HyperParameters,dump=False):
    parametrs= DataClass(PARAMETRS.DATA_NESTED)
    parametrs.convolutional.dilation_steps = hp.Int(
            'dilation_steps',
            default=4,
            min_value=3,
            max_value=5,
            step=1,
        )

    parametrs.transformer.share_weights = hp.Boolean(
            'share_weights',
            default=True,
        )

    dropout_rate: hp.Float(
        'dropout_rate',
        min_value=0,
        max_value=0.5,
        step=0.1,
    )

    choices = {'l2': 'l2', 'None': None}
    choice = hp.Choice(
        name='regularizer',
        values=list(choices),
        default='None',
    )
    parametrs.feed_forward.kernel_regularizer = choices[choice]

    choice = hp.Choice(
        name='out_activation',
        default='softmax',
        values=['softmax', 'none'],
    )
    choices={'softmax':'softmax','none':None}
    parametrs.feed_forward.out_activation = choices[choice]



    if dump: return hp
    return parametrs


# def configure(hp: keras_tuner.HyperParameters,dump=False):
#     parametrs= DataClass(PARAMETRS.DATA_NESTED)

#     parametrs.convolutional.dilation_steps = 5

#     parametrs.transformer.share_weights = False

#     choices = {'l2': 'l2', 'None': None}
#     choice = hp.Choice(
#         name='regularizer',
#         values=list(choices),
#         default='None',
#     )
#     parametrs.feed_forward.kernel_regularizer = choices[choice]

#     lr = hp.Choice(
#         name='lr',
#         default=0.0001,
#         values=[0.01, 0.001, 0.0005, 0.0001],
#     )
#     choices = {
#         'sgd':
#         tf.keras.optimizers.legacy.SGD(learning_rate=lr),
#         'rms':
#         tf.keras.optimizers.legacy.RMSprop(learning_rate=lr),
#         'adam':
#         tf.keras.optimizers.legacy.Adam(
#             learning_rate=lr,
#             beta_1=0.9,
#             beta_2=0.999,
#         ),
#     }
#     choice = hp.Choice(
#         name='optimazer',
#         default='adam',
#         values=['adam', 'rms', 'sgd'],
#     )
#     parametrs.optimizer = choices[choice]



#     if dump: return hp
#     return parametrs


# %%
## Build
def search_model(hp):
    parametrs = configure(hp)
    model = test_model.blocks.build_model(**parametrs.DATA_NESTED)
    return model


input_name = ''
while input_name == '':
    input_name = input(f"Input search name: ")
search_name = ModelBack.get_search_name(input_name)

print(
    f'Pattern model: {test_model.__name__}',
    f'Search name: {search_name}',
    'Parametrs:',
    configure(keras_tuner.HyperParameters()),
    sep='\n',
)

# %%
##Callbacks
callback_freq = 100
search_dir = f"{ModelBack.callback_path}/{search_name}"
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=search_dir,
        histogram_freq=callback_freq,
        update_freq=callback_freq,
    ),
]

to_dump = dump_config_function(configure)
to_dump.desc = input(f"Input description: ")

ModelBack.dump(to_dump, model_path=search_dir)
print(
    f"Callbacks:\n{[str(type(callback)).split('.')[-1] for callback in callbacks]}",
    f'Directory: {search_dir}',
    sep='\n',
)

# %%
## Build tuner
tuner = keras_tuner.GridSearch(
    hypermodel=search_model,
    objective="loss",
    executions_per_trial=1,
    directory=search_dir,
)

# %%
# %tensorboard

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

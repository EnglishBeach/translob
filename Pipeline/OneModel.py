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

from backend import DataBack,ModelBack,DataClass
seq_len = 100

# %%
from models import m_base as test_model

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
## Build
tf.keras.backend.clear_session()
restore = True if input('Restore? (y-yes, enter-no): ') == 'y' else False
input_name = ''
while input_name == '':
    input_name = input(
        f"Input train name to {'restore' if restore else 'build new'}: ")
if restore:
    model, train_name = ModelBack.restore_model(input_name)
else:
    ## Set up parametrs
    model = test_model.blocks.build_model(**DEFAULT_PARAMETRS.Data_nested)
    train_name = ModelBack.get_training_name(input_name)
    print(
        f'Pattern model: {test_model.__name__}',
        f'Train name: {train_name}',
        'Parametrs:',
        DEFAULT_PARAMETRS,
        sep='\n',
    )
model.summary()

# %%
## Callbacks
callback_freq = 'epoch'
train_dir = f'{ModelBack.callback_path}/{train_name}'

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=train_dir,
        histogram_freq=1,
        update_freq=callback_freq,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f'{train_dir}/checkpoints/' + '{epoch:04d}.keras',
        monitor="val_sp_acc",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq=callback_freq,
    )
]

print(
    f"Callbacks:\n{[str(type(callback)).split('.')[-1] for callback in callbacks]}",
    f'Directory: {train_dir}',
    sep='\n',
)


# %%
# %tensorboard

# %%
## Train
training_question = ''
while training_question not in ['y', 'n']:
    training_question = input(f'Start training now (y-yes) (n-exit): ')
if training_question == 'y':
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        callbacks=callbacks,
    )

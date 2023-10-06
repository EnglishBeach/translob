# # %%
# %tensorboard

# # %%
# ## For collab
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

from tools import data, utils
from models import m_base as test_model

seq_len = 100

# %%
## Save data


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
## Build
restore = True if input('Restore? (y-yes, enter-no): ') == 'y' else False
input_name = ''
date_tag = f'({datetime.datetime.now().strftime("%H-%M-%S--%d.%m")})'
while input_name == '':
    input_name = input(
        f"Input model name to {'restore' if restore else 'build new'}: ")

if restore:
    restore_path = f'{data.callback_path}/{input_name}/checkpoints'
    checkpoint_list = sorted(os.listdir(restore_path))
    model = tf.keras.models.load_model(f'{restore_path}/{checkpoint_list[-1]}')
    print(f'Model {checkpoint_list[-1]} loaded')
    input_name =\
        ('restore' if restore else '')+\
        (checkpoint_list[-1].split('.')[0])+\
        '_'+\
        (input_name.split('(')[0])
else:
    pars = utils.DataClass(test_model.PARAMETRS)
    model = test_model.build_model(**pars.Info_expanded)
    print('Model built')

model_name = f"{input_name}{date_tag}"
print(f'Model name: {model_name}')
model.summary()

# %%
## Callbacks
callback_freq = 'epoch'
model_dir = f'{data.callback_path}/{model_name}'
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=model_dir,
        histogram_freq=1,
        update_freq=callback_freq,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f'{model_dir}/checkpoints/' + '{epoch:04d}.keras',
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
    f'Model directory: {model_dir}',
    sep='\n',
)


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

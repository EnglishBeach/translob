# %%
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
import datetime
import numpy as np
import tensorflow as tf
import keras
from tools import data, utils

data.use_jupyter(False)
seq_len = 100

# %%
## Save data


# %%
## Load data
part = input('Data dole %? (press enter for all): ')
if part == '': part = 1
else: part = float(part) / 100

row_data = data.load_saved_datas(part)
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
from models import m_base as test_model

model_name = ''
while model_name == '':
    model_name = input('Training name: ')

pars = utils.DataClass(test_model.PARAMETRS)
model = test_model.build_model(**pars.Info_expanded)
print(model_name)
model.summary()

# %%
## Callbacks
callback_freq = 'epoch'
name_tag = datetime.datetime.now().strftime("%H-%M-%S--%d.%m")
log_dir = data.prefix+f'/Temp/callbacks/{model_name}({name_tag})'
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=callback_freq,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f"{log_dir}/checkPoints",
        monitor="val_sp_acc",
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq=callback_freq,
    )
]
print(callbacks, log_dir, sep='\n')

# %%
## Train
training_question = ''
while training_question not in ['y', 'n']:
    training_question = input('Start training now? (y-yes) (n-exit): ')
if training_question == 'y':
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        callbacks=callbacks,
    )

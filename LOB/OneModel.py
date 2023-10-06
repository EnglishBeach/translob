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
import datetime
import numpy as np
import tensorflow as tf
import keras
from tools import data, utils

data.check_using_jupyter()
seq_len = 100

# %%
## Save data


# %%
## Load data
part = input('Data proportion 100-0 in % (press enter for all): ')
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
restore = True if input('Restore? (y-yes, enter-no): ') == 'y' else False
input_name = ''
date_tag = f'({datetime.datetime.now().strftime("%H-%M-%S--%d.%m")})'
while input_name == '':
    input_name = input(
        f"Input model name to {'restore' if restore else 'build new'}: ")

if restore:
    restore_path = data.prefix + f'/Temp/callbacks/{input_name}/checkPoints'
    model = tf.keras.models.load_model(restore_path)
else:
    from models import m_base as test_model

    pars = utils.DataClass(test_model.PARAMETRS)
    model = test_model.build_model(**pars.Info_expanded)

model_name = f"{input_name}{'R' if restore else ''}{date_tag}"
print(f'Model name:{model_name}')
model.summary()

# %%
## Callbacks
callback_freq = 'epoch'
model_dir = data.prefix + f'/Temp/callbacks/{model_name}'
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=model_dir,
        histogram_freq=1,
        update_freq=callback_freq,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f"{model_dir}/checkpoint.tf",
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
    training_question = input(f'Start training now model:\n {model_name}\n(y-yes) (n-exit): ')
if training_question == 'y':
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        callbacks=callbacks,
    )

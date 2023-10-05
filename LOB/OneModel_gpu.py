# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm
import keras
import datetime

from utilities import data,dataclass


seq_len = 100

## Load data
if input('Quck training? (y-yes): ')=='y':
    data_len=2000
else:
    data_len=None

row_data = data.load_saved_datas(data_len)
# row_data = data.load_dataset(horizon=4)
data.inspect_datas(row_data)

datasets = data.build_datasets(
    datas=row_data,
    batch_size=100,
    seq_len=seq_len,
)
(ds_train, ds_val, ds_test) =\
(datasets['train'], datasets['val'], datasets['test'])
data.inspect_datasets(datasets)


from models import m_base as test_model

## Build
model_name=''
while model_name=='':
    model_name = input('Training name: ')

pars = dataclass.DataClass(test_model.PARAMETRS)
model = test_model.build_model(**pars.Info_expanded)
print(model_name)
model.summary()

# Callbacks
name_tag = datetime.datetime.now().strftime("%H-%M-%S--%d.%m")
log_dir = f'Temp/callbacks/{model_name}({name_tag})'
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=1,
    ),
    # tf.keras.callbacks.ModelCheckpoint(
    #     f"{save_path}/checkPoints",
    #     monitor="val_loss",
    #     verbose=0,
    #     save_best_only=False,
    #     save_weights_only=True,
    #     mode="auto",
    #     save_freq=50,
    #     options=None,
    #     initial_value_threshold=None,
    # )
]
print(callbacks, log_dir, sep='\n')

## Train
if input('Start training now? (y-yes): ')=='y':
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        callbacks=callbacks,
    )

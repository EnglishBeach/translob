import datetime
import keras
import tensorflow as tf

from tools import data, utils

seq_len = 100

## Load data
data_len = input('How much data need? (press enter for all): ')
if data_len == '': data_len = None
else: data_len = int(data_len)
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

## Build
from models import m_base as test_model

model_name = ''
while model_name == '':
    model_name = input('Training name: ')

pars = utils.DataClass(test_model.PARAMETRS)
model = test_model.build_model(**pars.Info_expanded)
print(model_name)
model.summary()

## Callbacks
callback_freq = 'epoch'
name_tag = datetime.datetime.now().strftime("%H-%M-%S--%d.%m")
log_dir = f'Temp/callbacks/{model_name}({name_tag})'
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=callback_freq,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f"{log_dir}/checkPoints",
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq=callback_freq,
        options=None,
        initial_value_threshold=None,
    )
]
print(callbacks, log_dir, sep='\n')

## Train
training_question = ''
while training_question not in ['y', 'n']:
    training_question = input('Start training now? (y-yes) (n-exit): ')
if training_question == 'y':
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        # callbacks=callbacks,
    )

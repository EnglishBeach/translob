# %%
import datetime
import numpy as np
import tensorflow as tf

from tools import utils,express
from tools.express import Connector
from models import m_base as test_model
seq_len = 100

Connector.connect()

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
parametrs = utils.DataClass(test_model.PARAMETRS)
parametrs

# %%
## Build
tf.keras.backend.clear_session()
restore = True if input('Restore? (y-yes, enter-no): ') == 'y' else False
input_name = ''
date_tag = f'({datetime.datetime.now().strftime("%H-%M-%S--%d.%m")})'
while input_name == '':
    input_name = input(
        f"Input train name to {'restore' if restore else 'build new'}: ")

if restore:
    model,train_name= express.restore_model(input_name)
else:
    # parametrs = utils.DataClass(test_model.PARAMETRS)

## Set up parametrs

    model = test_model.blocks.build_model(**parametrs.Data_nested)
    train_name = f"{input_name}{date_tag}"
    print(
    f'Pattern model: {test_model.__name__}',
    f'Train name: {train_name}',
    'Parametrs:',
    parametrs,
    sep='\n',
)
model.summary()

# %%
## Callbacks
callback_freq = 'epoch'
train_dir = f'{Connector.callback_path}/{train_name}'

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

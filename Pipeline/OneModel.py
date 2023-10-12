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
    project_name = 'LOB'
    from google.colab import drive
    drive.mount('/content/drive/', force_remount=True)
    os.chdir(f'/content/drive/My Drive/{project_name}/Pipeline')
    os.system('pip install automodinit keras_tuner')
    os.system(
        f'nohup /usr/bin/python3 /content/drive/MyDrive/{project_name}/Pipeline/Colab_saver.py'
    )


def kaggle_action():
    ...


platform = get_platform()
if platform == 'colab':
    colab_action()
elif platform == 'kaggle':
    kaggle_action()

from tools.express import Backend

Backend.set_paths(platform)

# %%
import datetime
import numpy as np
import tensorflow as tf
from tools import utils, express

from models import m_base as test_model

seq_len = 100

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
    model, train_name = express.restore_model(input_name)
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
train_dir = f'{Backend.callback_path}/{train_name}'

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

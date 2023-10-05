import tensorflow as tf
import datetime
import numpy as np
from tqdm import tqdm
import keras
import keras_tuner

import data
from utilites import DataClass

seq_len = 100

model_name = ''
while model_name == '':
    model_name = input('Search name: ')
model_name

# Datasets
if input('Quck training? (y-yes): ') == 'y':
    data_len = 2000
else:
    data_len = None

row_data = data.load_saved_datas(max_number=data_len)
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


# Tuner parametrs
def configure(hp: keras_tuner.HyperParameters):

    class CN_search(DataClass):
        dilation_steps = hp.Int(
            'cn_layers',
            default=4,
            min_value=3,
            max_value=5,
            step=1,
        )

    class AN_search(DataClass):
        attention_heads = 3
        blocks = 2
        share_weights = hp.Boolean(
            'share_weights',
            default=True,
        )

    class FF_search(DataClass):
        dropout_rate = hp.Float(
            'dropout_rate',
            default=0.1,
            min_value=0,
            max_value=0.5,
            step=0.1,
        )

    lr = hp.Float(
        'optimazer__lerarning_rate',
        default=1e-4,
        min_value=1e-6,
        max_value=1e-3,
    )
    beta1 = hp.Float(
        'optimazer__beta1',
        default=0.9,
        min_value=0.5,
        max_value=1.1,
        step=0.1,
    )
    beta2 = hp.Float(
        'optimazer_beta1',
        default=0.999,
        min_value=0.5,
        max_value=1.1,
        sampling='log',
    )

    class Full_search(DataClass):
        cn = CN_search()
        an = AN_search()
        ff = FF_search()
        optimizer = keras.optimizers.Adam(
            name="Adam",
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
        )

    return Full_search()


from models import m_base, m_preln


# Build model
def search_base_model(hp):
    hyper_pars_data = configure(hp)
    pars_data = DataClass(m_base.PARAMETRS)
    pars = pars_data.Info_expanded
    pars.update(hyper_pars_data.Info_expanded)

    model = m_base.build_model(**pars)
    return model


def search_m_preln(hp):
    hyper_pars_data = configure(hp)
    pars_data = DataClass(m_preln.PARAMETRS)
    pars = pars_data.Info_expanded
    pars.update(hyper_pars_data.Info_expanded)

    model = m_preln.build_model(**pars)
    return model


# Callbacks
log_dir = f'Temp/callbacks/{model_name}({datetime.datetime.now().strftime("%H-%M-%S--%d.%m")})'
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

# Build tuner
tuner = keras_tuner.GridSearch(
    hypermodel=search_base_model,
    objective="loss",
    executions_per_trial=1,
    directory=log_dir,
    # project_name='parametrs',
)

# Train
if input('Start training now? (y-yes)') == 'y':
    tuner.search(
        ds_train,
        validation_data=ds_val,
        epochs=20,
        callbacks=callbacks,
    )

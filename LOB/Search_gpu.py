import datetime
import keras
import keras_tuner
import tensorflow as tf

from tools import data
from tools.utils import DataClass

seq_len = 100

## Datasets
data_len = int(input('How much data need? (press enter for all): '))
if data_len == '': data_len = None
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


## Tuner parametrs
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


## Build model
from models import m_preln as test_model


def search_model(hp):
    hyper_pars_data = configure(hp)
    pars_data = DataClass(test_model.PARAMETRS)
    pars = pars_data.Info_expanded
    pars.update(hyper_pars_data.Info_expanded)

    model = test_model.build_model(**pars)
    return model


search_name = ''
while search_name == '':
    search_name = input('Search name: ')
search_name

##Callbacks
name_tag = datetime.datetime.now().strftime("%H-%M-%S--%d.%m")
log_dir = f'Temp/callbacks/search_{search_name}({name_tag})'
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch',
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f"{log_dir}/checkPoints",
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq='epoch',
        options=None,
        initial_value_threshold=None,
    )
]
print(callbacks, log_dir, sep='\n')

## Build tuner
tuner = keras_tuner.GridSearch(
    hypermodel=search_model,
    objective="loss",
    executions_per_trial=1,
    directory=log_dir,
)

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

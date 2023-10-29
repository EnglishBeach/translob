import os as _os
import datetime as _datetime
import numpy as _np
import tensorflow as _tf
import keras_tuner as _keras_tuner
import json as _json
import pathlib as _pathlib
from sklearn.model_selection import train_test_split as _train_test_split

from tensorflow.keras.utils import timeseries_dataset_from_array as _timeseries_dataset_from_array

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649


class DataClass:
    """
    make only lover case parametrs and not start with _
    All this methods (exept __call__) only for beauty representation :)
    """

    @staticmethod
    def __not_data(field=None, get=False, not_data_fields: set = set()):
        if not get:
            not_data_fields.add(field.__name__)
            return field
        else:
            return not_data_fields

    def __init__(
        self,
        target_dict: dict = {},
        name: str = '',
    ):
        for field_name in self.__get_all_fields():
            field = getattr(self, field_name)
            if type(self.__init__) == type(field):
                # TODO: add signature
                field_result = field.__func__
                setattr(self, field_name, field_result)

    def __new__(
        cls,
        target_dict: dict = {},
        name: str = '',
    ):
        """
        build from nested dict
        """
        if target_dict != {}:
            result = DataClass()
            return result.__rec_build(name, target_dict)
        return super().__new__(cls)

    def __rec_build(self, field_name: str, field):
        if not isinstance(field, dict):
            setattr(self, field_name, field)
            return None

        result = DataClass()
        setattr(self, field_name, result)

        for inner_field_name, inner_field in field.items():
            inner_result = result.__rec_build(
                inner_field_name,
                inner_field,
            )
            if inner_result is not None:
                setattr(self, field_name, inner_result)
        return result

    def __call__(self, **kwargs: dict):
        """
        Set up parametrs
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __get_all_fields(self):
        # Add except fields
        options = list(
            filter(
                lambda x:
                (x[0] != '_') and (x not in self.__not_data(get=True)),
                dir(self),
            ))
        return options

    def __repr__(self) -> str:
        """
        Representation of options
        """
        return self.__rec_print()[4:]

    def _rec_print_depr(self, self_margin: str = ''):
        if not isinstance(self, DataClass):
            return f'{self}'

        result = self_margin
        for field_name in self.__get_all_fields():
            inner_result = DataClass._rec_print_depr(
                getattr(self, field_name),
                self_margin + ' ' * 4,
            )
            result += f'\n{self_margin}{field_name}: {inner_result}'

        if self_margin == '':
            return result[1:]
        else:
            return result

    def __rec_print(
        self,
        self_name: str = '',
        self_header: str = '',
        last=True,
    ):
        end = "└─ "
        pipe = "│  "
        tee = "├─ "
        blank = "   "

        result = ''
        if not isinstance(self, DataClass):
            result = self
            if '<' in repr(self):
                result = repr(self).split('at')[0].replace('<', '').strip()

            return f'{self_header}{end if last else tee}{self_name}: {result}\n'

        result = f'{self_header}{end if last else tee}{self_name}\n'
        fields = self.__get_all_fields()
        for field_name in fields:
            inner_result = DataClass.__rec_print(
                getattr(self, field_name),
                self_name=field_name,
                self_header=f'{self_header}{blank if last else pipe}',
                last=field_name == fields[-1])

            result += inner_result[6:]

        return result

    @property
    @__not_data
    def DATA_NESTED(self):
        """
        Containing options dict
        """
        return self.__rec_nested()

    def __rec_nested(self, self_name=None):
        if not isinstance(self, DataClass):
            return {self_name: self}

        result = {}
        for field_name in self.__get_all_fields():
            inner_result = DataClass.__rec_nested(
                getattr(self, field_name),
                field_name,
            )
            result.update(inner_result)

        if self_name is None:
            return result
        else:
            return {self_name: result}

    @property
    @__not_data
    def DATA_EXPANDED(self):
        return {
            compound_key.strip()[2:]: value
            for value, compound_key in self.__rec_expanded()
        }

    def __rec_expanded(self, composite_key=''):
        if not isinstance(self, DataClass):
            yield (self, composite_key)
        else:
            for field_name in self.__get_all_fields():
                for inner_result in DataClass.__rec_expanded(
                        getattr(self, field_name),
                        str(composite_key) + '__' + str(field_name),
                ):
                    yield inner_result

    def __getitem__(self, value):
        if isinstance(value, list | tuple):
            result = {}
            for i in value:
                result.update({i: getattr(self, i, None)})
            return DataClass(result)
        result = getattr(self, value, None)
        if isinstance(result, DataClass):
            return DataClass(result.DATA_NESTED)
        else:
            return result


def set_backend(platform):
    workdirs = {
        'python': '.',
        'jupyter': '..',
        'kaggle': '',
        'colab': '..',
    }

    workdir = workdirs[platform]
    ModelBack.callback_path = workdir + ModelBack.callback_path
    DataBack.dataset_path = workdir + DataBack.dataset_path

    print(
        f'Dataset  : {DataBack.dataset_path}',
        f'Callbacks: {ModelBack.callback_path}',
        sep='\n',
    )


class DataBack:
    dataset_path = r'/dataset/saved_data'
    last_data_info ={}
    @staticmethod
    def process_data(
        data,
        *,
        seq_len,
        batch_size=128,
    ) -> _tf.data.Dataset:
        x: _np.ndarray = data[0]
        y: _np.ndarray = data[1]

        def set_shape(value_x: _np.ndarray, value_y: _np.ndarray):
            value_x.set_shape((None, seq_len, x.shape[-1]))
            return value_x, value_y

        ds = _timeseries_dataset_from_array(
            data=x,
            targets=y,
            batch_size=batch_size,
            sequence_length=seq_len,
        )

        return ds.map(set_shape)

    # TODO: to child class
    def process_raw_data(self, raw_data, *, horizon=1):
        # 40 == 10 price + volume asks + 10 price + volume bids
        x = raw_data[:40, :].T

        y = raw_data[-5 + horizon, :].T
        return [x[:-1], (y[1:] - 1).astype(_np.int32)]  # shift y by 1

    # def process_raw_data(self, raw_data, **process_parametrs):
    #     raise NotImplementedError

    def read_raw_data_files(
        self,
        train_paths=[],
        test_paths=[],
        **process_parametrs,
    ):
        self.last_data_info = dict(
            read_from='raw_data',
            train_paths=train_paths,
            test_paths=test_paths,
            process_parametrs=process_parametrs,
        )
        print('Read raw data, info writen to last_data_info.')

        train = []
        for path in train_paths:
            train_data = _np.loadtxt(path)
            train.append(self.process_raw_data(
                    train_data,
                    **process_parametrs,
                ))

        test = []
        for path in test_paths:
            test_data = _np.loadtxt(path)

            test.append(self.process_raw_data(
                test_data,
                **process_parametrs,
            ))
        return train, test

    def read_saved_data(
        self,
        proportion=1,
        train_indexes: list = [],
        val_indexes: list = [],
        test_indexes: list = [],
    ):
        self.last_data_info = dict(
            read_from='saved',
            proportion=proportion,
            train_indexes=train_indexes,
            val_indexes=val_indexes,
            test_indexes=test_indexes,
        )
        print('Read saved data, info writen to last_data_info.')

        restrict = lambda x: x[:int(len(x) * proportion)]
        train = []
        for i in train_indexes:
            with open(f'{self.dataset_path}/x_train{i}.npy', 'rb') as file:
                x = _np.load(file)
                x = restrict(x)

            with open(f'{self.dataset_path}/y_train{i}.npy', 'rb') as file:
                y = _np.load(file)
                y = restrict(y)
            train.append((x, y))

        val = []
        for i in val_indexes:
            with open(f'{self.dataset_path}/x_val{i}.npy', 'rb') as file:
                x = _np.load(file)
                x = restrict(x)
            with open(f'{self.dataset_path}/y_val{i}.npy', 'rb') as file:
                y = _np.load(file)
                y = restrict(y)
            val.append((x, y))

        test = []
        for i in test_indexes:
            with open(f'{self.dataset_path}/x_test{i}.npy', 'rb') as file:
                x = _np.load(file)
                x = restrict(x)
            with open(f'{self.dataset_path}/y_test{i}.npy', 'rb') as file:
                y = _np.load(file)
                y = restrict(y)
            test.append((x, y))

        return train, val, test

    @staticmethod
    def validation_split(data, shuffle=False, val_size=0.2):
        train, val = [], []
        for x, y in data:
            x_train, x_val, y_train, y_val = _train_test_split(
                x, y, test_size=val_size, shuffle=shuffle)
            train.append([x_train, y_train])
            val.append([x_val, y_val])
        return train, val

    def save_data(self, *, train=None, val=None, test=None):
        if train is not None:
            for i, (x, y) in enumerate(train):
                _np.save(file=f"{self.dataset_path}/x_train{i}.npy", arr=x)
                _np.save(file=f"{self.dataset_path}/y_train{i}.npy", arr=y)

        if val is not None:
            for i, (x, y) in enumerate(val):
                _np.save(file=f"{self.dataset_path}/x_val{i}.npy", arr=x)
                _np.save(file=f"{self.dataset_path}/y_val{i}.npy", arr=y)

        if test is not None:
            for i, (x, y) in enumerate(test):
                _np.save(file=f"{self.dataset_path}/x_test{i}.npy", arr=x)
                _np.save(file=f"{self.dataset_path}/y_test{i}.npy", arr=y)


    def data_to_dataset(self, data, **process_dataset_kwargs):
        result: _tf.data.Dataset = None
        for i, data in enumerate(data):
            ds = self.process_data(data, **process_dataset_kwargs)
            if i != 0:
                result = result.concatenate(ds)
            else:
                result = ds
        return result

    @staticmethod
    def inspect_data(**datas):
        print('    Datas:')
        for name, data in datas.items():
            for x, y in data:
                print(
                    f'{name: <10}: x= {str(x.shape): <15} | y= {str(y.shape): <15}'
                )

    @staticmethod
    def inspect_dataset(**datas):
        print('    Datasets:')
        for name, dataset in datas.items():
            print(
                f'{name: <6}: {[len(dataset)]+ list(dataset.element_spec[0].shape)[1:]}'
            )

class ModelBack:
    callback_path = '/Temp/callbacks'

    @staticmethod
    def _get_time_tag():
        time_now = _datetime.datetime.now(
            _datetime.timezone.utc) + _datetime.timedelta(hours=3)
        return f'({time_now.strftime("%H;%M;%S--%d.%m")})'

    @classmethod
    def get_search_name(cls, name):
        return f'search_{name}{cls._get_time_tag()}'

    @classmethod
    def get_training_name(cls, name):
        return f'{name}{cls._get_time_tag()}'

    @classmethod
    def restore_model(cls, input_name):
        restore_path = f"{cls.callback_path}/{input_name}/checkpoints"
        checkpoint_list = sorted(_os.listdir(restore_path))

        model = _tf.keras.models.load_model(
            f'{restore_path}/{checkpoint_list[-1]}')

        restored_epoch = checkpoint_list[-1].split('.')[0]
        restored_name = input_name.split('(')[0]
        new_name = f'restore_{restored_epoch}_{restored_name}{cls._get_time_tag()}'

        print(f'Model {input_name} restored on  {restored_epoch} ')
        return model, new_name

    @classmethod
    def dump_data(cls, data: DataClass, model_path):
        _pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        params_str = str(data.DATA_NESTED)
        params_dict = eval(params_str.replace('<', "'<").replace('>', ">'"))
        with open(f'{model_path}/descriprion.json', 'w') as file:
            _json.dump(params_dict, file, indent=2)


def dump_config_function(configure_function):

    def hyper_dump(var):
        var_type = var['class_name']
        default = var['config']['default']

        if var_type in ['Int', 'Float']:
            diap = f"[{var['config']['min_value']} : {var['config']['max_value']}]"
            steps = f"{var['config']['sampling']} {var['config']['step']}"
            values = f"{diap} by {steps}"
        elif var_type == 'Choice':
            values = var['config']['values']
        elif var_type == 'Boolean':
            values = 'True/False'
        else:
            values = ''

        result = {'type': var_type}
        result.update({
            'values': values,
            'default': var['config']['default'],
            'conditions': var['config']['conditions'],
        })

        return var['config']['name'], result

    configurated_params = configure_function(_keras_tuner.HyperParameters(),
                                             dump=True)
    hp_config = configurated_params.get_config()

    param_dataclass = DataClass(
        dict([hyper_dump(variable) for variable in hp_config['space']]))
    return param_dataclass

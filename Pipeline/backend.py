import os as _os
import numpy as _np
import tensorflow as _tf
from sklearn.model_selection import train_test_split as _train_test_split

from tensorflow.keras.utils import timeseries_dataset_from_array as _timeseries_dataset_from_array

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649


def set_backend(platform):
    workdirs = {
        'python': '.',
        'jupyter': '..',
        'kaggle': '',
        'colab': '..',
    }

    workdir = workdirs[platform]
    Models.callback_path = workdir + Models.callback_path
    Datasets.dataset_path = workdir + Datasets.dataset_path

    print(
        f'Dataset  : {Datasets.dataset_path}',
        f'Callbacks: {Models.callback_path}',
        sep='\n',
    )


# Paths
class Datasets:
    dataset_path = r'/saved_data'

    @staticmethod
    def process_dataset(
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

    @staticmethod
    def process_data(data, *, horizon=1):
        # 40 == 10 price + volume asks + 10 price + volume bids
        x = data[:40, :].T

        y = data[-5 + horizon, :].T
        return [x[:-1], (y[1:] - 1).astype(_np.int32)]  # shift y by 1

    @classmethod
    def from_files(
        cls,
        train_files=[],
        test_files=[],
        **process_data_kwargs,
    ):
        train = []
        val = []
        for path in train_files:
            train_data = _np.loadtxt(path)
            train.append(cls.process_data(train_data, **process_data_kwargs))

        test = []
        for path in test_files:
            test_data = _np.loadtxt(path)

            test.append(cls.process_data(test_data, **process_data_kwargs))
        return train, test

    @classmethod
    def _one_from_saved(cls, kind, i, proportion):
        x_path = f"{cls.paths['dataset']}/x_{kind}{i}.npy"
        with open(x_path, 'rb') as file:
            x = _np.load(file)
        y_path = f"{cls.paths['dataset']}/y_{kind}{i}.npy"
        with open(y_path, 'rb') as file:
            y = _np.load(file)

        assert (x != None) or (y != None), FileExistsError(
            f'File on {x_path} or {y_path} not exists')

        data_len = int(len(x) * proportion)
        x = x[:data_len]
        y = y[:data_len]

        return x, y

    @classmethod
    def from_saved(
        cls,
        proportion=1,
        train_indexes: list = [],
        val_indexes: list = [],
        test_indexes: list = [],
    ):
        restrict = lambda x: x[:int(len(x) * proportion)]
        train = []
        for i in train_indexes:
            with open(f'{cls.dataset_path}/x_train{i}.npy', 'rb') as file:
                x = _np.load(file)
                x = restrict(x)

            with open(f'{cls.dataset_path}/y_train{i}.npy', 'rb') as file:
                y = _np.load(file)
                y= restrict(y)
            train.append((x, y))

        val = []
        for i in val_indexes:
            with open(f'{cls.dataset_path}/x_val{i}.npy', 'rb') as file:
                x = _np.load(file)
                x = restrict(x)
            with open(f'{cls.dataset_path}/y_val{i}.npy', 'rb') as file:
                y = _np.load(file)
                y= restrict(y)
            val.append((x, y))

        test = []
        for i in test_indexes:
            with open(f'{cls.dataset_path}/x_test{i}.npy', 'rb') as file:
                x = _np.load(file)
                x = restrict(x)
            with open(f'{cls.dataset_path}/y_test{i}.npy', 'rb') as file:
                y = _np.load(file)
                y= restrict(y)
            test.append((x, y))

        return train, val, test

    @staticmethod
    def validation_split(data, shuffle=True, val_size=0.2):
        train, val = [], []
        for x, y in data:
            x_train, x_val, y_train, y_val = _train_test_split(
                x, y, test_size=val_size, shuffle=shuffle)
            train.append([x_train, y_train])
            val.append([x_val, y_val])
        return train, val

    @classmethod
    def save_data(cls, *, train=None, val=None, test=None):
        if train is not None:
            for i, (x, y) in enumerate(train):
                _np.save(file=f"{cls.dataset_path}/x_train{i}.npy", arr=x)
                _np.save(file=f"{cls.dataset_path}/y_train{i}.npy", arr=y)

        if val is not None:
            for i, (x, y) in enumerate(val):
                _np.save(file=f"{cls.dataset_path}/x_val{i}.npy", arr=x)
                _np.save(file=f"{cls.dataset_path}/y_val{i}.npy", arr=y)

        if test is not None:
            for i, (x, y) in enumerate(test):
                _np.save(file=f"{cls.dataset_path}/x_test{i}.npy", arr=x)
                _np.save(file=f"{cls.dataset_path}/y_test{i}.npy", arr=y)

    @classmethod
    def build_dataset(cls, data, **process_dataset_kwargs):
        result = {}
        result: _tf.data.Dataset = None

        for i, data in enumerate(data):
            ds = cls.process_dataset(data, **process_dataset_kwargs)
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


class Models:
    callback_path = f'/Temp/callbacks'

    @classmethod
    def restore_model(cls, input_name):
        restore_path = f"{cls.callback_path}/{input_name}/checkpoints"
        checkpoint_list = sorted(_os.listdir(restore_path))

        model = _tf.keras.models.load_model(
            f'{restore_path}/{checkpoint_list[-1]}')

        print(f'Model {checkpoint_list[-1]} loaded')
        restored_epoch = checkpoint_list[-1].split('.')[0]
        restored_name = input_name.split('(')[0]
        new_name = f'restore_{restored_epoch}_{restored_name}'

        return model, new_name

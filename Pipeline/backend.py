import os as _os
import numpy as _np
import tensorflow as _tf

from tensorflow.keras.utils import timeseries_dataset_from_array as _timeseries_dataset_from_array

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649


# Paths
class Dataset:
    _workdir = {
        'python': '.',
        'jupyter': '..',
        'kaggle': '',
        'colab': '..',
    }
    _callback_path = f'/Temp/callbacks'
    _dataset_path = r'/saved_data'

    @staticmethod
    def process_dataset(
        data,
        *,
        seq_len,
        batch_size=128,
    ):
        x: _np.ndarray = data[0]
        y: _np.ndarray = data[1]

        def set_shape(value_x, value_y):
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
    def _set_paths(cls, platform):
        cls.workdir_path = cls._workdir[platform]
        cls.callback_path = cls.workdir_path + cls._callback_path
        cls.dataset_path = cls.workdir_path + cls._dataset_path

        print(
            f'Callbacks: {cls.callback_path}',
            f'Dataset  : {cls.dataset_path}',
            sep='\n',
        )

    @classmethod
    def get_paths(cls):
        return dict(
            callback_path=cls.callback_path,
            dataset_path=cls.dataset_path,
        )

    @classmethod
    def load_data_from_files(
        cls,
        train_files=[],
        test_files=[],
        piece=1,
        val_proportion=0.8,
        **process_data_kwargs,
    ):
        train = []
        val = []
        for path in train_files:
            data = _np.loadtxt(path)
            need_len = int(data.shape[1] * piece)
            data = data[:, :need_len]

            train_data = data[:, :int(data.shape[1] * val_proportion)]
            val_data = data[:, int(data.shape[1] * val_proportion):]

            train.append(cls.process_data(train_data, **process_data_kwargs))
            val.append(cls.process_data(val_data, **process_data_kwargs))

        test = []
        for path in test_files:
            test_data = _np.loadtxt(path)
            need_len = int(test_data.shape[1] * piece)
            test_data = test_data[:, :need_len]

            test.append(
                cls.process_data(test_data,**process_data_kwargs))
        return train, val, test

    @classmethod
    def _load_one_from_saved(self, kind, i, proportion):
        with open(f"{Dataset.paths['dataset']}/x_{kind}{i}.npy", 'rb') as file:
            x = _np.load(file)
        with open(f"{Dataset.paths['dataset']}/y_{kind}{i}.npy", 'rb') as file:
            y = _np.load(file)

            data_len = int(len(x) * proportion)
            x = x[:data_len]
            y = y[:data_len]

        return x, y

    @classmethod
    def load_data_from_saved():
        ...


def restore_model(input_name):
    restore_path = f"{Dataset.paths['callback']}/{input_name}/checkpoints"
    checkpoint_list = sorted(_os.listdir(restore_path))

    model = _tf.keras.models.load_model(
        f'{restore_path}/{checkpoint_list[-1]}')

    print(f'Model {checkpoint_list[-1]} loaded')
    restored_epoch = checkpoint_list[-1].split('.')[0]
    restored_name = input_name.split('(')[0]

    return model, f'restore_{restored_epoch}_{restored_name}'

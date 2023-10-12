import os
import numpy as _np
import tensorflow as tf

from .utils import inspect_data, inspect_dataset, build_dataset

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649


# Paths
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

    print(f'{platform} platform used')

    return platform


def path_correct(platform):
    global prefix, save_path, dataset_path, callback_path
    prefix= {
        'python':'.',
        'jupyter':'..',
        'kaggle':'',
        'colab':'',
    }
    prefix = prefix[platform]

    save_path = prefix + r'/LOB/saved_data'
    callback_path = prefix + f'/Temp/callbacks'

    dataset_path = prefix + r'/dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore'


path_correct(get_platform())


# Save
def save_data(x, y, name):
    """
    kinds = 'test', 'train', 'val'
    """
    with open(f'{save_path}/x_{name}.npy', 'wb') as file:
        _np.save(file, x)
    with open(f'{save_path}/y_{name}.npy', 'wb') as file:
        _np.save(file, y)


# Load data
def _gen_data(data, horizon):
    x = data[:40, :].T  # 40 == 10 price + volume asks + 10 price + volume bids
    y = data[-5 + horizon, :].T  # 5
    return [x[:-1], (y[1:] - 1).astype(_np.int32)]  # shift y by 1


def load_datas(horizon):
    dec_data = _np.loadtxt(
        f'{dataset_path}_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')

    dec_train = dec_data[:, :int(_np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(_np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = _np.loadtxt(
        f'{dataset_path}_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')

    dec_test2 = _np.loadtxt(
        f'{dataset_path}_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')

    dec_test3 = _np.loadtxt(
        f'{dataset_path}_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')

    dec_test = _np.hstack((dec_test1, dec_test2, dec_test3))

    datas = {
        'train': _gen_data(dec_train, horizon),
        'val': _gen_data(dec_val, horizon),
        'test': _gen_data(dec_test, horizon),
    }
    return datas


def load_saved_datas(part=1):
    """
    kinds = 'test', 'train', 'val'
    """
    datas = {}
    for kind in ['train', 'val', 'test']:
        try:
            with open(f'{save_path}/x_{kind}.npy', 'rb') as file:
                x = _np.load(file)
            with open(f'{save_path}/y_{kind}.npy', 'rb') as file:
                y = _np.load(file)
            data_len = int(len(x) * part)
            x = x[:data_len]
            y = y[:data_len]

        except FileNotFoundError:
            x, y = None, None

        datas.update({kind: [x, y]})
    return datas


def inspect_datas(datas: dict):
    print('    Datas:')
    for name in datas:
        data = datas[name]
        inspect_data(data, name)


def build_datasets(datas: dict, batch_size, seq_len):
    datasets = {}
    for kind in datas:
        data = datas.get(kind, None)
        ds = None
        if data is not None:
            ds = build_dataset(
                x=data[0],
                y=data[1],
                batch_size=batch_size,
                seq_len=seq_len,
            )
        datasets.update({kind: ds})

    return datasets


def inspect_datasets(datasets: dict):
    print('    Datasets:')
    for name in datasets:
        ds = datasets[name]
        inspect_dataset(ds, name)


def restore_model(input_name):
    restore_path = f'{callback_path}/{input_name}/checkpoints'
    checkpoint_list = sorted(os.listdir(restore_path))

    model = tf.keras.models.load_model(f'{restore_path}/{checkpoint_list[-1]}')

    print(f'Model {checkpoint_list[-1]} loaded')
    restored_epoch = checkpoint_list[-1].split('.')[0]
    restored_name = input_name.split('(')[0]

    return model, f'restore_{restored_epoch}_{restored_name}'

import numpy as _np
from tensorflow.keras.utils import timeseries_dataset_from_array as _timeseries_dataset_from_array
from .utils import inspect_data, inspect_dataset

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

# Paths
def check_using_jupyter():
    try:
        get_ipython().__class__.__name__
        using_jupyter=True
    except NameError:
        using_jupyter=False

    global prefix,save_path,dataset_path,callback_path
    prefix = '..' if using_jupyter else '.'
    save_path = prefix + r'/LOB/saved_data'
    dataset_path = prefix + r'/dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore'
    callback_path =prefix + f'/Temp/callbacks'

check_using_jupyter()


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
            data_len = int(len(x)*part)
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


def build_dataset(
    x: _np.ndarray,
    y: _np.ndarray,
    seq_len,
    batch_size=128,
    **timeseries_kwargs,
):

    def set_shape(value_x, value_y):
        value_x.set_shape((None, seq_len, x.shape[-1]))
        return value_x, value_y

    ds = _timeseries_dataset_from_array(
        data=x,
        targets=y,
        batch_size=batch_size,
        sequence_length=seq_len,
        **timeseries_kwargs,
    )

    return ds.map(set_shape)


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


if __name__ == '__main__':
    a = load_saved_datas()
    print(a)

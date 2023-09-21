import numpy as np
from keras.preprocessing import timeseries_dataset_from_array as _timeseries_dataset_from_array

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
_FI2010_DIR_ = r'D:\WORKS\translob\dataset\BenchmarkDatasets'
_add_path_ = r'/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore'

PATH = _FI2010_DIR_ + _add_path_


def _gen_data(data, horizon):
    x = data[:40, :].T  # 40 == 10 price + volume asks + 10 price + volume bids
    y = data[-5 + horizon, :].T  # 5
    return x[:-1], (y[1:] - 1).astype(np.int32)  # shift y by 1


def load_dataset(horizon):
    dec_data = np.loadtxt(
        f'{PATH}_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')

    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt(
        f'{PATH}_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')

    dec_test2 = np.loadtxt(
        f'{PATH}_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')

    dec_test3 = np.loadtxt(
        f'{PATH}_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')

    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    result = (
        _gen_data(dec_train, horizon),
        _gen_data(dec_val, horizon),
        _gen_data(dec_test, horizon),
    )
    return result


def load_saved_data():
    """
    kinds = 'test', 'train', 'val'
    """
    data = []
    for kind in ['train', 'val', 'test']:
        try:
            with open(f'saved_data/x_{kind}.npy', 'rb') as file:
                x = np.load(file)
            with open(f'saved_data/y_{kind}.npy', 'rb') as file:
                y = np.load(file)
        except FileNotFoundError:
            x, y = None, None
        data.append((x, y))
    return data


def save_data(x, y, name):
    """
    kinds = 'test', 'train', 'val'
    """
    with open(f'saved_data/x_{name}.npy', 'wb') as file:
        np.save(file, x)
    with open(f'saved_data/y_{name}.npy', 'wb') as file:
        np.save(file, y)


def build_dataset(
    x: np.ndarray,
    y: np.ndarray,
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
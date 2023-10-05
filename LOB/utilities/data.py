import numpy as np
from tensorflow.keras.utils  import timeseries_dataset_from_array as _timeseries_dataset_from_array

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
_FI2010_DIR_ = r'D:\WORKS\translob\dataset\BenchmarkDatasets'
_add_path_ = r'/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore'

DATASET_PATH = _FI2010_DIR_ + _add_path_
NUMPY_DATA_PATH =r'D:\WORKS\translob\LOB\saved_data'

# Save
def save_data(x, y, name):
    """
    kinds = 'test', 'train', 'val'
    """
    with open(f'LOB/saved_data/x_{name}.npy', 'wb') as file:
        np.save(file, x)
    with open(f'LOB/saved_data/y_{name}.npy', 'wb') as file:
        np.save(file, y)


# Load data
def _gen_data(data, horizon):
    x = data[:40, :].T  # 40 == 10 price + volume asks + 10 price + volume bids
    y = data[-5 + horizon, :].T  # 5
    return [x[:-1], (y[1:] - 1).astype(np.int32)]  # shift y by 1


def load_datas(horizon):
    dec_data = np.loadtxt(
        f'{DATASET_PATH}_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')

    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt(
        f'{DATASET_PATH}_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')

    dec_test2 = np.loadtxt(
        f'{DATASET_PATH}_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')

    dec_test3 = np.loadtxt(
        f'{DATASET_PATH}_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')

    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    datas = {
        'train': _gen_data(dec_train, horizon),
        'val': _gen_data(dec_val, horizon),
        'test': _gen_data(dec_test, horizon),
    }
    return datas


def load_saved_datas(max_number=None):
    """
    kinds = 'test', 'train', 'val'
    """
    datas = {}
    for kind in ['train', 'val', 'test']:
        try:
            with open(f'{NUMPY_DATA_PATH}/x_{kind}.npy', 'rb') as file:
                x = np.load(file)
            with open(f'{NUMPY_DATA_PATH}/y_{kind}.npy', 'rb') as file:
                y = np.load(file)
            if max_number is not None:
                x = x[:max_number]
                y = y[:max_number]
        except FileNotFoundError:
            x, y = None, None

        datas.update({kind: [x, y]})
    return datas


def inspect_data(data, name='data'):
    if data is not None:
        x = data[0]
        y = data[1]
        print(f'{name: <10}: x= {str(x.shape): <15} | y= {str(y.shape): <15}')
    else:
        print(f'{name <10}: None')


def inspect_datas(datas: dict):
    print('    Datas:')
    for name in datas:
        data = datas[name]
        inspect_data(data, name)


# build datasets
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


def inspect_dataset(ds, name='dataset'):
    if ds is not None:
        print(f'{name: <10}: {[len(ds)]+ list(ds.element_spec[0].shape)[1:]}')
    else:
        print(f'{name <10}: None')


def inspect_datasets(datasets: dict):
    print('    Datasets:')
    for name in datasets:
        ds = datasets[name]
        inspect_dataset(ds, name)

if __name__=='__main__':
    a = load_saved_datas()
    print(a)

import numpy as np

from . import utils

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649


# Save
def save_data(x, y, name, path=r'./LOB/saved_data'):
    """
    kinds = 'test', 'train', 'val'
    """
    with open(f'{path}/x_{name}.npy', 'wb') as file:
        np.save(file, x)
    with open(f'{path}/y_{name}.npy', 'wb') as file:
        np.save(file, y)


# Load data
def _gen_data(data, horizon):
    x = data[:40, :].T  # 40 == 10 price + volume asks + 10 price + volume bids
    y = data[-5 + horizon, :].T  # 5
    return [x[:-1], (y[1:] - 1).astype(np.int32)]  # shift y by 1


def load_datas(
    horizon,
    path=r'./dataset/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore',
):
    dec_data = np.loadtxt(
        f'{path}_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')

    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt(
        f'{path}_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')

    dec_test2 = np.loadtxt(
        f'{path}_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')

    dec_test3 = np.loadtxt(
        f'{path}_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')

    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    datas = {
        'train': _gen_data(dec_train, horizon),
        'val': _gen_data(dec_val, horizon),
        'test': _gen_data(dec_test, horizon),
    }
    return datas


def load_saved_datas(
    max_number=None,
    path=r'./LOB/saved_data',
):
    """
    kinds = 'test', 'train', 'val'
    """
    datas = {}
    for kind in ['train', 'val', 'test']:
        try:
            with open(f'{path}/x_{kind}.npy', 'rb') as file:
                x = np.load(file)
            with open(f'{path}/y_{kind}.npy', 'rb') as file:
                y = np.load(file)
            if max_number is not None:
                x = x[:max_number]
                y = y[:max_number]
        except FileNotFoundError:
            x, y = None, None

        datas.update({kind: [x, y]})
    return datas


def inspect_datas(datas: dict):
    print('    Datas:')
    for name in datas:
        data = datas[name]
        utils.inspect_data(data, name)


def build_datasets(datas: dict, batch_size, seq_len):
    datasets = {}
    for kind in datas:
        data = datas.get(kind, None)
        ds = None
        if data is not None:
            ds = utils.build_dataset(
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
        utils.inspect_dataset(ds, name)


if __name__ == '__main__':
    a = load_saved_datas()
    print(a)

import numpy as np

# download FI2010 dataset from
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
FI2010_DIR = r'D:\WORKS\translob\dataset\BenchmarkDatasets'


def gen_data(data, horizon):
    x = data[:40, :].T  # 40 == 10 price + volume asks + 10 price + volume bids
    # FIXME: delete .T
    y = data[-5 + horizon, :].T  # 5
    return x[:-1], (y[1:] - 1).astype(np.int32)  # shift y by 1


def load_dataset(horizon):

    dec_data = np.loadtxt(
        f'{FI2010_DIR}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt'
    )

    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt(
        f'{FI2010_DIR}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt'
    )

    dec_test2 = np.loadtxt(
        f'{FI2010_DIR}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt'
    )

    dec_test3 = np.loadtxt(
        f'{FI2010_DIR}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt'
    )

    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))



    result = (
        gen_data(dec_train,horizon),
        gen_data(dec_val,horizon),
        gen_data(dec_test, horizon)) #yapf: disable

    return result

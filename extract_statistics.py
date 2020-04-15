import pickle
import scipy
import pandas as pd
import numpy as np


def coeff_channel(csr_matrix, chan1, chan2):
    """
    Computing mean correlation coefficient by upper triangle matrix

    Args:
        csr_matrix(sparce or dense matrix )

    Return:
        corr(float): mean correlation coefficient overall channels
        :param chan2: (int) channel id 1
        :param chan1: (int) channel 1d 2
    """
    size = csr_matrix.shape[0]
    mtx = csr_matrix.todense()
    mtx = mtx + mtx.T - np.diag(np.diag(mtx))
    mean_r = mtx[chan1, chan2]
    return mean_r


def mean_coeff_channel(csr_matrix, chan_id):
    """
    Computing mean correlation coefficient by upper triangle matrix

    Args:
        csr_matrix(sparce or dense matrix )

    Return:
        corr(float): mean correlation coefficient overall channels
        :param chan_id:
    """
    size = csr_matrix.shape[0]
    mtx = csr_matrix.todense()
    mtx = mtx + mtx.T - 2 * np.diag(np.diag(mtx))
    mean_r = np.sum(mtx[chan_id, :]) / (size - 1)
    return mean_r


def mean_coeff(csr_matrix):
    """
    Computing mean correlation coefficient by upper triangle matrix

    Args:
        csr_matrix(sparce or dense matrix )

    Return:
        corr(float): mean correlation coefficient overall channels
    """
    size = csr_matrix.shape[0]
    mean_r = (2 * np.sum(csr_matrix) - size) / (size * (size - 1))
    return mean_r


def compute_diff(data, func, state1='state2', state2='state3', cortype="pli", **kwargs):
    """
    computing difference between specified states
    Args
        data(dict): dictionary with information about each person, be freqs
        func (function): function which could process csr matrix
        state1 (string):
        state2 (string):
        cortype (string): ispc or pli
        channels(string or number from 1 to 19)

    Return
        df (pandas dataframe)

    """
    names = list(data.keys())
    num_freqs = len(data[names[0]][state1])
    colnames = [f"f_{data[names[0]][state1][f][0]}" for f in range(num_freqs)]
    tidx = 1 if cortype == "ispc" else 2
    values = []
    for name in names:
        if len(kwargs) == 0:
            row_value = [func(data[name][state2][f][tidx]) - func(data[name][state1][f][tidx]) for f in
                         range(num_freqs)]
        else:
            row_value = [
                func(data[name][state2][f][tidx], **kwargs) - func(data[name][state1][f][tidx], **kwargs)
                for f in range(num_freqs)]
        values.append(row_value)
    return pd.DataFrame(values, columns=colnames, index=names)




if __name__ == '__main__':
    with open('ispc_all.pickle', 'rb') as handle:
        ispc_all = pickle.load(handle)
    #df_ispc = compute_diff(ispc_all, mean_coeff)
    df_ispc = compute_diff(ispc_all, mean_coeff_channel, chan_id = 3)
    #df_ispc = compute_diff(ispc_all, coeff_channel, chan1=1, chan2=2)






    print('ok')

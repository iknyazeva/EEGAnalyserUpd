from RestingEEG import RestingEEG
import matplotlib.pyplot as plt
import os


def test_load_data(filepath, plot_show=True):
    exp = RestingEEG(filepath)

    exp.load_data(plot_show)

    return exp


def test_GaussianFilter(filepath, f, fwhm):
    exp = RestingEEG(filepath)
    exp.load_data(plot_show=False)
    exp.data = exp.data[:, :10000]
    filtdata = exp.filterFGx(f, fwhm)
    return filtdata


def test_phase_synch_two_series(filepath, f, fwhm, chan1, chan2):
    exp = RestingEEG(filepath)
    exp.load_data(plot_show=False)
    exp.data = exp.data[:, :10000]
    rho_ispc, rho_pli, rho_pow = exp.phase_sync_two_series(f, fwhm, chan1, chan2)
    return rho_ispc, rho_pli, rho_pow



def test_wavelet_phase_sync_two_series(filepath, f, chan1, chan2):
    exp = RestingEEG(filepath)
    exp.load_data(plot_show=False)
    exp.data = Exp.data[:, :10000]
    rho_ispc, rho_pli, rho_pow = exp.test_wavelet_phase_sync_two_series(filepath, f, chan1, chan2)
    return rho_ispc, rho_pli, rho_pow


def test_wavelet_phase_synch_all(filepath, f):
    exp = RestingEEG(filepath)
    exp.load_data(plot_show=False)
    exp.data = exp.data[:, :10000]
    rho_ispc, rho_pli, rho_pow = exp.wavelet_phase_synch_all(f)

    return rho_ispc, rho_pli, rho_pow


def test_phase_sync_all(filepath, f, fwhm):
    exp = RestingEEG(filepath)
    exp.load_data(plot_show=False)
    exp.data = exp.data[:, :10000]
    rho_ispc, rho_pli, rho_pow = exp.phase_sync_all(f, fwhm)

    return rho_ispc, rho_pli, rho_pow






if __name__ == "__main__":
    folder_path = "../Data_Txt"
    filename = "afl_01.TXT"
    f = 30
    fwhm = 3
    filepath = os.path.join(folder_path, filename)

    #fnames = os.listdir(folder_path)
    rho_ispc, rho_pli, rho_pow = test_phase_sync_all(filepath, f, fwhm)
    rho_ispc_w, rho_pli_w, rho_pow_w =  test_wavelet_phase_synch_all(filepath, f)

    print(os.path.isfile(filepath))

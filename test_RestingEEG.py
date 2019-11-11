from RestingEEG import RestingEEG
import matplotlib.pyplot as plt
import os

datapath = "/Users/Irisha1/MyProjects/EEG/RestingEEG/Data_Txt"
filename  = os.path.join(datapath,'afl_01.TXT')

filepath  = os.path.join(datapath, filename)

def test_load_data(filepath, plot_show = True):


    Exp = RestingEEG(filepath)

    data = Exp.load_data(plot_show)

    return  data


def test_filter(filepath, f,fwhm):
    Exp = RestingEEG(filepath)
    Exp.load_data(plot_show = False)
    filtdata = Exp.filterFGx(f,fwhm)
    return filtdata


def test_corrmat(filepath, f, fwhm):
    Exp = RestingEEG(filepath)
    Exp.load_data(plot_show=False)
    Exp.data = Exp.data[:, :100000]
    corrmat = Exp.phase_sync_all(f,fwhm)
    return corrmat


def test_pow_corrmat(filepath, f, fwhm):
    Exp = RestingEEG(filepath)
    Exp.load_data(plot_show=False)
    Exp.data = Exp.data[:, :100000]
    corrmat = Exp.pow_sync_all(f, fwhm)
    return corrmat



if __name__ == "__main__":
    #data = test_load_data(filepath,plot_show=True)
    f  = 30
    fwhm = 3
    corrmat = test_corrmat(filepath, f, fwhm)
    plt.subplot(121);plt.matshow(corrmat, fignum=False)
    corrmat = test_pow_corrmat(filepath, f, fwhm)
    plt.subplot(122); plt.matshow(corrmat,fignum=False)

    #filtdata = Exp.filterFGx(f, fwhm)
    #angldata = Exp.compute_angles(f,fwhm)
    #plt.figure(figsize = (10,5))
    ##plt.subplot(121); plt.plot(Exp.data[0, :1000])
    #rho = Exp.phase_sync_two_series(f, fwhm, 1, 2)
    #plt.subplot(121); plt.plot(filtdata[1, :1000])
    #plt.subplot(122); plt.plot(angldata[1, :1000])
    plt.show()
__author__ = 'Irina Knyazeva'
from RestingEEG import RestingEEG
from RestingEEG import  compute_relative_power
import matplotlib.pyplot as plt
import numpy as np
import os

datapath = "/Users/Irisha1/MyProjects/EEG/Kuperin/00_Data_EEG_Meditation/DataTxt"
filename  = os.path.join(datapath,'Bax1_med_130.TXT')
filename_fon  = os.path.join(datapath,'Bax1_fon_120.TXT')

srate = 250
def wspectrum_med_to_fon(filename, filename_fon, elList = [0,1], num_freqs = 10):

    Meds = RestingEEG(filename, srate=srate, num_freqs = num_freqs)

    Meds.load_data(plot_show = False)

    Fon = RestingEEG(filename_fon, srate=250, num_freqs = num_freqs)
    Fon.load_data(plot_show=False)
    pow_fon = np.absolute(Fon.wavelet_transform(elList)**2)

    pow = np.absolute(Meds.wavelet_transform(elList)**2)

    return pow, pow_fon, Meds



def compute_coher(filename, chanel1_id, chanel2_id, srate=srate, num_freqs = 10):
    Meds = RestingEEG(filename, srate, num_freqs)
    Meds.load_data(plot_show=False)
    spc = Meds.spectral_coher(chanel1_id=chanel1_id, chanel2_id=chanel2_id)


    return spc




def compute_norm(filename, filename_fon):
    pow, pow_fon, Meds = wspectrum_med_to_fon(filename, filename_fon)
    mean_vector = np.mean(pow_fon, axis =1)
    norm_pow = 10 * np.log10(pow/ mean_vector[:, None])
    timeExp = np.arange(Meds.data.shape[1])*1/srate

    plt.contourf(timeExp, Meds._freqs, norm_pow[:,:,0], cmap=plt.cm.jet)
    plt.contourf(timeExp, Meds._freqs, norm_pow[:,:,1], cmap=plt.cm.jet )

    plt.show()


def plot_spectrum(chan1,chan2):

    spc = compute_coher(filename, chan1, chan2)
    spc_fon = compute_coher(filename_fon, chan1, chan2)

    times = spc['times'] * 1 / srate

    plt.subplot(211);
    plt.contourf(times, spc['freqs'], spc['spc'], cmap=plt.cm.jet);
    plt.colorbar();
    plt.title('Meditation, spc for channel {} and {}'.format(spc['chan_dict'][chan1], spc['chan_dict'][chan2]))
    plt.subplot(212);
    plt.contourf(times, spc_fon['freqs'], spc_fon['spc'], cmap=plt.cm.jet);
    plt.colorbar();
    plt.title('Resting, spc for channel {} and {}'.format(spc['chan_dict'][chan1], spc['chan_dict'][chan2]))

    plt.show()


def plot_wavelet_pow(filename, filename_fon):
    relPow = compute_relative_power(filename,filename_fon)
    return relPow

if __name__ == "__main__":
    relPow = plot_wavelet_pow(filename, filename_fon)
    times = relPow['times'] * 1 / srate
    id = 1
    plt.contourf(times, relPow['freqs'], relPow['pow'][:,:,id], cmap=plt.cm.jet);
    plt.title('Relative power spectrum for channel {} '.format(relPow['chan_dict'][relPow['chans'][id]]))
    plt.show()






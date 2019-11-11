__author__ = 'Irina Knyazeva'
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
from scipy.signal import hilbert


class RestingEEG:


    def __init__(self, filename, srate  = 500, num_chan = 19):

        """

        :param filename:  filename with the records, should be text file with the values in columns
        :param srate: sampling rate, default 500Hz
        :param num_chan: number of channels, 19 in this case
        """
        self.filename = filename
        self.srate = srate
        self.num_chan = 19
        self.data = []
        self.chan_id = 'all'

    def load_data(self, plot_show = True):
        """
        todo:         добавить соответствие между номерами

        :return:

        """
        data = np.loadtxt(self.filename)
        if plot_show:
            plt.figure(figsize=(10, 20))
            for i in range(self.num_chan):
                plt.subplot(self.num_chan, 1, i + 1)
                plt.plot(data[:, i])
            plt.show()
        self.data = data[:,:19].T
        return data


    def filterFGx(self,f,fwhm, chan_id = 'all'):
        """
        Narrow-band filter via frequency-domain Gaussian, could be applied to the data

        :return: filter_data
        """
        hz = np.linspace(0, self.srate,self.data.shape[1])
        s = fwhm * (2 * np.pi - 1) / (4 * np.pi)
        x = hz - f
        fx = np.exp(-0.5*np.power(x/s,2))
        fx = fx/np.max(fx)
        self.chan_id = chan_id
        if chan_id == 'all':

            filtdata = 2 * np.real(ifft(fft(self.data, axis = 1)*fx, axis = 1))
        else:
            filtdata = 2 * np.real(ifft(fft(self.data[chan_id,:].reshape(1,-1), axis=1) * fx, axis=1)).reshape(1,-1)

        return filtdata


    def compute_angles(self,f,fwhm):
        filtdata = self.filterFGx(f,fwhm,self.chan_id)
        angldata = np.angle(hilbert(filtdata, axis = 1))

        return angldata

    def phase_sync_two_series(self,f,fwhm, chan1, chan2):
        filtdata1 = self.filterFGx(f, fwhm, chan_id=chan1)
        filtdata2  =self.filterFGx(f, fwhm, chan_id=chan2)
        angldata1 = np.angle(hilbert(filtdata1, axis=1))
        angldata2 = np.angle(hilbert(filtdata2, axis=1))
        rho = np.abs(np.mean(np.exp(1j*(angldata1-angldata2))))
        return rho

    def phase_sync_all(self,f,fwhm):


        corrmat = np.zeros((self.num_chan,self.num_chan))
        filtdata = self.filterFGx(f, fwhm, chan_id='all')
        for i in range(self.num_chan):
            for j in range(i+1,self.num_chan):
                angldata1 = np.angle(hilbert(filtdata[i,:]))
                angldata2 = np.angle(hilbert(filtdata[j,:]))

                corrmat[i,j] = np.abs(np.mean(np.exp(1j * (angldata1 - angldata2))))
        corrmat = corrmat+corrmat.T
        np.fill_diagonal(corrmat,1)

        return corrmat

    def pow_sync_all(self, f, fwhm):

        filtdata = self.filterFGx(f, fwhm, chan_id='all')

        powdata = np.abs(hilbert(filtdata, axis = 1))
        corrmat = np.corrcoef(powdata)






        return corrmat


__author__ = 'Irina Knyazeva'

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.signal import hilbert


class RestingEEG:

    def __init__(self, filename, srate=500, num_chan=19, min_freq=3, max_freq=30, num_freqs=10):

        """

        :param filename:  filename with the records, should be text file with the values in columns
        :param srate: sampling rate, default 500Hz
        :param num_chan: number of channels, 19 in this case
        :param min_freq: minimum considered frequency
        :param max_freq: maximum considered frequency
        """
        self.filename = filename
        self.srate = srate
        self.num_chan = num_chan
        self.data = []
        self.chan_id = 'all'
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self._freqs = np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), self.num_freqs)
        self.chan_dict = {0: 'Fp1-Ref', 1: 'Fp2', 2: 'F7', 3: 'F3', 4: 'Fz', 5: 'F4', 6: 'F8', 7: 'T3', 8: 'C3',
                          9: 'Cz', 10: 'C4',
                          11: 'T4', 12: 'T5', 13: 'P3', 14: 'Pz', 15: 'P4', 16: 'T6', 17: 'O1', 18: 'O2'}

    def load_data(self, plot_show=True):
        """
        todo:         добавить соответствие между номерами

        :return: object with the EEG data

        """
        data = np.loadtxt(self.filename)
        if plot_show:
            plt.figure(figsize=(10, 20))
            for i in range(self.num_chan):
                plt.subplot(self.num_chan, 1, i + 1)
                plt.plot(data[:, i])
            plt.show()
        self.data = data[:, :19].T

    def filterFGx(self, f, fwhm, chan_id='all'):
        """
        Narrow-band filter via frequency-domain Gaussian, could be applied to the data
        :param f: central frequency
        :param fwhm: full width half maximum  - width of Gaussian filter

        :return: filter_data at specified frequency
        """
        hz = np.linspace(0, self.srate, self.data.shape[1])
        s = fwhm * (2 * np.pi - 1) / (4 * np.pi)
        x = hz - f
        fx = np.exp(-0.5 * np.power(x / s, 2))
        fx = fx / np.max(fx)
        self.chan_id = chan_id
        if chan_id == 'all':

            filtdata = 2 * np.real(ifft(fft(self.data, axis=1) * fx, axis=1))
        else:
            filtdata = 2 * np.real(
                ifft(fft(self.data[chan_id, :].reshape(len(chan_id), -1), axis=1) * fx, axis=1)).reshape(len(chan_id),
                                                                                                         -1)

        return filtdata

    def compute_angles(self, f, fwhm):

        filtdata = self.filterFGx(f, fwhm, self.chan_id)
        angldata = np.angle(hilbert(filtdata, axis=1))

        return angldata

    def phase_sync_two_series(self, f, fwhm, chan1, chan2):
        """

        :param f: central frequency
        :param fwhm: full width half maximum  - width of Gaussian filter, scalar
        :param chan1: id of chanel, integer
        :param chan2: id of second chanel, integer
        :return:
        """
        #filter data with the specified frequency
        filtdata = self.filterFGx(f, fwhm, chan_id=[chan1, chan2])

        #compute angles for each chanel
        complex_data = hilbert(filtdata)
        angldata1 = np.angle(complex_data[0, :])
        angldata2 = np.angle(complex_data[1, :])
        #phase difference
        phase_diff = np.exp(1j * (angldata1 - angldata2))
        #inter site phase coherence
        rho_ispc = np.abs(np.mean(phase_diff))
        #phase locked index
        rho_pli = np.abs(np.mean(np.sign(np.imag(phase_diff))))
        rho_pow = np.corrcoef(np.abs(complex_data))[0][1]
        return rho_ispc, rho_pli, rho_pow

    def wavelet_conv(self, f, chanid="all"):
        """
        perform wavelet convolution with the specified frequency, another way for filtering data

        :param f: frequence for Morlet
        :param chanid: list of ids chanels, list of int
        :return: convolved data, complex one
        """

        wavtime = np.arange(-1.5, 1.5 + 1 / self.srate, 1 / self.srate)
        half_wave = np.floor((len(wavtime) - 1) / 2).astype(int)
        range_cycles = [4, 8]
        freqs = np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), 20)
        slist = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), 20)
        idx = (np.abs(freqs - f)).argmin()
        s = slist[idx] / (2 * np.pi * f)
        wavelet = np.exp(2 * 1j * np.pi * f * wavtime) * np.exp(-np.power(wavtime, 2) / (2 * np.power(s, 2)))
        nWave = wavtime.shape[0]
        nData = self.data.shape[1]
        nConv = nWave + nData - 1

        waveletX = fft(wavelet, nConv)
        waveletX = waveletX / np.max(waveletX)
        if chanid == "all":
            dataX = fft(self.data, nConv)
        else:
            dataX = fft(self.data[chanid], nConv)

        convData = ifft(waveletX * dataX)
        convData = convData[:, half_wave:-half_wave]
        return convData

    def wavelet_phase_synch_all(self, f, ispc=True, pli = True, power = True):
        """

        :param f: central frequency
        :param ispc: bool
        :param pli: bool
        :param power: bool
        :return:  matrix with ispc, pli and power coeffs
        """

        convData = self.wavelet_conv(f)

        ispc_corrmat = np.zeros((self.num_chan, self.num_chan))
        pli_corrmat = np.zeros((self.num_chan, self.num_chan))
        power_corrmat =  np.zeros((self.num_chan, self.num_chan))

        for i in range(self.num_chan):
            for j in range(i + 1, self.num_chan):
                phase_diff = np.exp(1j * (np.angle(convData[i, :]) - np.angle(convData[j, :])))

                if ispc:
                    ispc_corrmat[i, j] = np.abs(np.mean(phase_diff))
                if pli:
                    pli_corrmat[i, j] = np.abs(np.mean(np.sign(np.imag(phase_diff))))

        ispc_corrmat = ispc_corrmat + ispc_corrmat.T
        np.fill_diagonal(ispc_corrmat, 1)
        pli_corrmat = pli_corrmat + pli_corrmat.T
        np.fill_diagonal(pli_corrmat, 1)

        if power:
            powdata = np.abs(convData)
            power_corrmat = np.corrcoef(powdata)

        return ispc_corrmat, pli_corrmat, power_corrmat

    def wavelet_phase_sync_two_series(self, f, chan1, chan2):

        convData = self.wavelet_conv(f, chanid=[chan1, chan2])
        angldata1 = np.angle(convData[0, :])

        angldata2 = np.angle(convData[1, :])

        phase_diff = np.exp(1j * (angldata1 - angldata2))
        rho_ispc = np.abs(np.mean(phase_diff))
        rho_pli = np.abs(np.mean(np.sign(np.imag(phase_diff))))
        rho_pow = np.corrcoef(np.abs(convData))[0][1]
        return rho_ispc, rho_pli, rho_pow

    def phase_sync_all(self, f, fwhm,ispc = True, pli = True, power = True):

        ispc_corrmat = np.zeros((self.num_chan, self.num_chan))
        pli_corrmat = np.zeros((self.num_chan, self.num_chan))
        power_corrmat = np.zeros((self.num_chan, self.num_chan))

        filtdata = self.filterFGx(f, fwhm, chan_id='all')
        for i in range(self.num_chan):
            for j in range(i + 1, self.num_chan):
                angldata1 = np.angle(hilbert(filtdata[i, :]))
                angldata2 = np.angle(hilbert(filtdata[j, :]))
                phase_diff = np.exp(1j * (angldata1 - angldata2))
                if ispc:
                    ispc_corrmat[i, j] = np.abs(np.mean(phase_diff))
                if pli:
                    pli_corrmat[i, j] = np.abs(np.mean(np.sign(np.imag(phase_diff))))


        ispc_corrmat = ispc_corrmat + ispc_corrmat.T
        np.fill_diagonal(ispc_corrmat, 1)
        pli_corrmat = pli_corrmat + pli_corrmat.T
        np.fill_diagonal(pli_corrmat, 1)
        if power:
            powdata = np.abs(hilbert(filtdata, axis=1))
            power_corrmat = np.corrcoef(powdata)

        return ispc_corrmat, pli_corrmat, power_corrmat

    def pow_sync_all(self, f, fwhm):

        filtdata = self.filterFGx(f, fwhm, chan_id='all')

        powdata = np.abs(hilbert(filtdata, axis=1))
        corrmat = np.corrcoef(powdata)

        return corrmat

    def wavelet_transform(self, elList=None, print_=True):

        """
        Function for wavelet transform computing


        elList: list of electrodes, for ex. [0,2,3], if null  - compute all electrodes
        return wt: wavelet transformation for time series or matrix

        freq: vector of used frequencies in Hz
        srate: sampling rate in Hz
        min_freq: minimal frequency of wavelet
        max_freq: max frequency of wavelet
        num_freq: number of frequency in interval from min to max

        """

        assert self.data.shape[0] > 0, "You need to define EEG data for wavelet transform first"
        assert self.data.shape[0] == self.num_chan, "First dimension must be equal to number of channels"

        if elList:
            data = np.take(self.data, elList, axis=0).reshape(len(elList), -1)
        else:
            data = self.data

        wt = np.zeros(shape=(self.num_freqs, self.data.shape[1], data.shape[0]), dtype=complex)

        range_cycles = [4, 8]
        s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), self.num_freqs) / (
                2 * np.pi * self._freqs)
        wavtime = np.arange(-1, 1 + 1 / self.srate, 1 / self.srate)
        half_wave = np.floor((len(wavtime) - 1) / 2).astype(int)

        nWave = wavtime.shape[0]

        nData = data.shape[1]
        nConv = nWave + nData - 1

        if elList == None:
            elList = list(range(0, data.shape[0]))

        for i, el in enumerate(elList):
            if print_:
                print("Compute channel: ", el + 1)
            dataX = fft(data[i, :], nConv)
            for fi in range(0, self.num_freqs):
                wavelet = np.exp(2 * 1j * np.pi * self._freqs[fi] * wavtime) * np.exp(
                    -np.power(wavtime, 2) / (2 * np.power(s[fi], 2)))
                waveletX = fft(wavelet, nConv)

                convData = ifft(waveletX * dataX)
                convData = convData[half_wave:-half_wave]
                wt[fi, :, i] = convData

        return wt

    def spectral_coher(self, chanel1_id, chanel2_id, num_points=300):

        """
        compute spectral coherence by wavelet transformed data
        """

        spectcoher = np.zeros(shape=(self.num_freqs, num_points), dtype=float)
        wt = self.wavelet_transform(elList=[chanel1_id, chanel2_id])
        data1 = wt[:, :, 0]
        data2 = wt[:, :, 1]
        # length of time window for phase averaging from 1.5 for lowest freq to 3 cycles
        timewindow = np.linspace(15, 18, self.num_freqs)
        # length of the largest time window in points
        time_window_largest = np.floor((1000 / self._freqs[0]) * timewindow[0] / (1000 / self.srate)).astype(int)
        times2saveidx = np.linspace(time_window_largest, self.data.shape[1] - time_window_largest, num_points).astype(
            int)

        for fi in range(0, self.num_freqs):
            sig1 = data1[fi, :]
            sig2 = data2[fi, :]

            spec1 = abs(sig1 * np.conj(sig1))
            spec2 = abs(sig2 * np.conj(sig2))
            # cross_spec = np.power(abs(sig1 * np.conj(sig2)), 2)
            cross_spec = sig1 * np.conj(sig2)
            # Averaging in the sliding window
            # compute time window in indices for this frequency and average inside the window
            time_window_idx = np.floor((1000 / self._freqs[fi]) * timewindow[fi] / (1000 / self.srate)).astype(int)
            for ti in range(0, len(times2saveidx)):
                Cxy = np.power(
                    abs(np.mean(cross_spec[times2saveidx[ti] - time_window_idx:times2saveidx[ti] + time_window_idx])),
                    2)
                Cxx = np.mean(spec1[times2saveidx[ti] - time_window_idx:times2saveidx[ti] + time_window_idx])
                Cyy = np.mean(spec2[times2saveidx[ti] - time_window_idx:times2saveidx[ti] + time_window_idx])
                spectcoher[fi, ti] = Cxy / (Cxx * Cyy)
        return {'spc': spectcoher, 'chanIds': [chanel1_id, chanel2_id], 'times': times2saveidx, 'freqs': self._freqs,
                'chan_dict': self.chan_dict}

    # def phase_coherence(self):


def compute_relative_power(filename, filename_fon, srate=250, num_freqs=10, num_points=300, elList=[0, 1]):
    stateEEG = RestingEEG(filename, srate=srate, num_freqs=num_freqs)

    stateEEG.load_data(plot_show=False)

    refEEG = RestingEEG(filename_fon, srate=srate, num_freqs=num_freqs)
    refEEG.load_data(plot_show=False)
    if elList:
        relPow = np.zeros(shape=(num_freqs, num_points, len(elList)), dtype=float)

    else:
        relPow = np.zeros(shape=(num_freqs, num_points, stateEEG.num_chan), dtype=float)

    # length of time window in number of cycles for lowest frequency
    num_cycles = 15
    time_window = np.floor((1000 / stateEEG._freqs[0]) * num_cycles / (1000 / stateEEG.srate)).astype(int)
    times2saveidx = np.linspace(time_window, stateEEG.data.shape[1] - time_window, num_points).astype(int)

    stateWT = stateEEG.wavelet_transform(elList=elList)
    fonWT = refEEG.wavelet_transform(elList=elList)
    mean_vector = np.mean(np.power(abs(fonWT), 2), axis=1)
    for ti in range(0, len(times2saveidx)):
        statePow = np.mean(
            np.power(abs(stateWT[:, times2saveidx[ti] - time_window:times2saveidx[ti] + time_window, :]), 2), axis=1)
        relPow[:, ti, :] = 10 * np.log10(statePow / mean_vector)

    return {'pow': relPow, 'times': times2saveidx, 'freqs': stateEEG._freqs, 'chan_dict': stateEEG.chan_dict,
            'chans': elList}

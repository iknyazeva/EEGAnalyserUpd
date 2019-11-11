__author__ = 'Irisha1'

import numpy as np
from EEGProcessing import EEGAnalyser
import pickle

myExperiment = EEGAnalyser(srate = 500,min_freq = 4, max_freq=30, num_freq=15)

myExperiment.freq = np.logspace(np.log10(myExperiment.min_freq),np.log10(myExperiment.max_freq),myExperiment.num_freq)
myExperiment.num_trials = 22
myExperiment.min_length = 3600


#Получаем нормализованные данные (все пробы от 0 до минимальной длины, 3600 здесь) и считаем вейвлет преобразование для всех каналов
compute_wavelet = False
waveFilename = 'WTImD0000418.pickle'
if (compute_wavelet):
    data = myExperiment.norm_data
    myExperiment.wavelet_transform(data)
    with open(waveFilename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(myExperiment.wt, f, pickle.HIGHEST_PROTOCOL)

else:
    with open(waveFilename, 'rb') as f:
        myExperiment.wt= pickle.load(f)


#myExperiment.phase_coherence(0,1)
myExperiment.spectral_coher(0,1)

myExperiment.phase_syncr['ispcs'].shape
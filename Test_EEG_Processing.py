__author__ = 'Irisha1'


import numpy as np
from EEGProcessing import EEGAnalyser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle




dir_name = '/Users/Irisha1/PycharmProjects/InteractiveJupiter/EEG/DATA/'
fileInfoPath = dir_name+'EEG_MentalInfo.pickle'


with open(fileInfoPath, 'rb') as f:
    EEG_info = pickle.load(f)



Im_names = EEG_info['list_Im']
k = 0
sig_number = 'D0000'+str(Im_names[k])
sigName = dir_name+'CSV/'+ sig_number +'.csv'
timingName = dir_name + 'timing/'+sig_number+'.TXT'
failedTrialsName = dir_name + 'failed_trials/'+sig_number+'.txt'
print(sigName,failedTrialsName,timingName)

myExperiment = EEGAnalyser(min_freq = 4, max_freq=30, num_freq=5)


print(myExperiment.srate)
#print(myExperiment.min_freq)

def test_timing():
    myExperiment = EEGAnalyser(srate=500, min_freq=4, max_freq=30, num_freq=5)

    timingPath = dir_name + 'timing/'
    min_trial_lens = myExperiment.find_minimal_trial(timingPath)
    print(myExperiment.min_length)

myExperiment.min_length = 3480


    #Загружаем сигнал
data = myExperiment.load_data(sigName, timingName, failedTrialsName)
    #sig = load_data(sigName, timingName, failedTrialsName, srate = 500).as_matrix()
#data.iloc[:,0].plot()
data = myExperiment.normalize_data()

ERP = myExperiment.computeERP()

freq,wt = myExperiment.wavelet_transform(elList=[0,3])

normwt = myExperiment.baseline_normalization(trial_average=False)

#print(wt.shape)



myExperiment.phase_coherence(0,3)
myExperiment.wt.shape
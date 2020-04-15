from contextlib import contextmanager
from RestingEEG import RestingEEG
import os
import glob
import scipy
import pickle


@contextmanager
def load_experiment(path):
    try:
        exp = RestingEEG(path)
        exp.load_data(plot_show=False)
        yield exp
    except OSError:
        print("We had an error with loading data!")


def compute_ispc_ind(folder_path, freqs=[3, 6, 9, 12, 16, 25, 35]): #central freqs for different rythms
    path = os.path.join(folder_path, "*.TXT")
    filelist = glob.glob(path)
    names = list(set([file.split('/')[-1].split('_')[0] for file in filelist]))
    results = dict()

    for id_, name in enumerate(names):
        print(f"Process patient number {id_} with the codename {name} \n")
        results[name] = dict([(f"state{i}", []) for i in range(1, 4)])
        for i in range(1, 4):
            filepath = f"{folder_path}{name}_0{i}.TXT"
            with load_experiment(filepath) as exp:
                for f in freqs:
                    ispc, pli, _ = exp.wavelet_phase_synch_all(f)
                    results[name][f"state{i}"].append((f, scipy.sparse.triu(ispc), scipy.sparse.triu(pli)))

    with open('ispc_all.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("ok")
    return results




if __name__ == '__main__':
    folder_path = "../Data_Txt/"
    #filename = "afl_01.TXT"
    results = compute_ispc_ind(folder_path)
    # filepath = os.path.join(folder_path, filename)

    # with load_experiment(filepath) as exp:
    #     ispc_corrmat, pli_corrmat, power_corrmat = exp.wavelet_phase_synch_all(f)

    print('ok')

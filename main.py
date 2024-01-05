import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
import csv
import pickle
import os
import dictionary
import pandas as pd
from loaddata.time_series import *
import datetime
import matchingpursuit
import matplotlib.pyplot as plt
import SVD
import time
from matchingpursuit import *
from SVD import *

def gen_dict(wavelet, resolution):
    file_path = f'/home/daniela/PycharmProjects/tesis/{wavelet}'
    if os.path.exists(file_path):
        print(f'The file "{file_path}" exists in the system.')
        with open(file_path, 'rb') as file:
            dict_ = pickle.load(file)
            #dict_ = file.read()
            return dict_

    else:
        print(f'The file "{file_path}" does not exist in the system.')
        # j = int(np.log2(resolution))
        numpy_dictionary = dictionary.const_dict_continue(wavelet, resolution)    # It's a numpy array
        with open(wavelet, 'wb') as file:
            pickle.dump(numpy_dictionary, file)
            return numpy_dictionary

def gen_dict_x(wavelet, resolution):
    file_path = f'/home/daniela/PycharmProjects/tesis/x{wavelet}'
    if os.path.exists(file_path):
        print(f'The file "{file_path}" exists in the system.')
        with open(file_path, 'rb') as file:
            dict_ = pickle.load(file)
            #dict_ = file.read()
            return dict_

    else:
        print(f'The file "{file_path}" does not exist in the system.')
        # j = int(np.log2(resolution))
        numpy_dictionary = dictionary.const_dict_continue_x(wavelet, resolution)    # It's a numpy array
        with open(f'x{wavelet}', 'wb') as file:
            pickle.dump(numpy_dictionary, file)
            return numpy_dictionary

def main(wavelet, resolution):
    dicc_y = gen_dict(wavelet, resolution)
    dicc_x = gen_dict_x(wavelet, resolution)

    # LOAD DATA
    data = read_ts_from_ibdb('AAPL', '1 day', None, '2023-08-31', last=1000)
    data_adj_close = data[0]['adj_close'][:resolution].to_numpy()

    ### Domain interpolation for the given delta
    x_discreto = np.arange(resolution)
    x_continuo = dicc_x[:, 0]
    for k in range(1, 21):
        aux_x = dicc_x[:, 0][1:] + 3*k
        x_continuo = np.concatenate((x_continuo, aux_x))

    # y_continuo = np.interp(x_continuo, x_discreto, data_adj_close)


    # ## SIGNAL PLOTS
    # plt.scatter(np.arange(64), data_adj_close, label='discrete')
    # plt.legend()
    # plt.show()
    # plt.plot(x_continuo, y_continuo, label='Signal')
    # plt.legend()
    # plt.show()

    # y_signal = y_continuo - y_continuo[0]
    # plt.plot(x_continuo, y_signal, label='Signal')
    # plt.legend()
    # plt.show()

    ## EXPERIMENTO ATOMS ALETORIOS
    atoms, j_s, a_s = random_atoms(dicc_y, dicc_x, x_continuo)
    ## INTERNAL PRODUCTS
    for i in range(5):
        for k in range(5):
            print('p_interno', np.dot(atoms[i], atoms[k])/1024, 'atomo1', j_s[i], a_s[i],
                  'atomo2',j_s[k], a_s[k])
    rango = range(1, 101)
    numeros_aleatorios = random.sample(rango, 5)
    numeros_aleatorios.sort(reverse=True)
    print('números_aleatorios', numeros_aleatorios)
    signal = np.zeros(64513)
    for i in range(5):
        signal += numeros_aleatorios[i]*atoms[i]

    ## MATHING PURSUIT FOR THE RANDOM EXPERIMENT
    r, ap = mp(dicc_y, dicc_x, signal, x_continuo, 10)
    #
    # plt.plot(x_continuo, signal, 'red', label='SIGNAL')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(x_continuo, ap, 'blue', label='APPROXIMATION')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main('db2',64)

    # data = read_ts_from_ibdb('AAPL', '1 day', None, '2023-08-31', last=1000)
    # data_adj_close = data[0]['adj_close'][:64].to_numpy()
    # print(data_adj_close)
    # print(type(data_adj_close))
    # print(len(data_adj_close))


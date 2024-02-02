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
import matplotlib.pyplot as plt
import SVD
import time
from matchingpursuit import *
from SVD import *
import random
import sys
import concurrent.futures
import multiprocessing


def gen_dict(wavelet, resolution, level):
    # file_path = f'/home/daniela/PycharmProjects/tesis/{wavelet}'
    file_path = f'{wavelet}{level}'
    if os.path.exists(file_path):
        print(f'The file "{file_path}" exists in the system.')
        with open(file_path, 'rb') as file:
            dict_ = pickle.load(file)
            # dict_ = file.read()
            return dict_

    else:
        print(f'The file "{file_path}" does not exist in the system.')
        # j = int(np.log2(resolution))
        numpy_dictionary = dictionary.const_dict_continue(wavelet, resolution, level)    # It's a numpy array
        with open(file_path, 'wb') as file:
            pickle.dump(numpy_dictionary, file)
            return numpy_dictionary


def gen_dict_x(wavelet, resolution, level):
    # file_path = f'/home/daniela/PycharmProjects/tesis/x{wavelet}'
    file_path = f'x{wavelet}{level}'
    if os.path.exists(file_path):
        print(f'The file "{file_path}" exists in the system.')
        with open(file_path, 'rb') as file:
            dict_ = pickle.load(file)
            # dict_ = file.read()
            return dict_

    else:
        print(f'The file "{file_path}" does not exist in the system.')
        # j = int(np.log2(resolution))
        numpy_dictionary = dictionary.const_dict_continue_x(wavelet, resolution, level)    # It's a numpy array
        with open(f'x{wavelet}{level}', 'wb') as file:
            pickle.dump(numpy_dictionary, file)
            return numpy_dictionary


def labeler(complete_series, resolution, days_after):
    """
    Returns a subarray from complete_series of length resolution along with a label.

    Args:
        complete_series: The large array from which the subarray is selected.
        resolution: The length of the subarray to be selected.
        days_after: Missing Daniela.

    Returns:
         subarray: subarray of complete_series with length resolution.
         label (int): The label associated with the subarray. It will be 1 if the last
                      entry of the subarray is greater than or equal to the penultimate entry,
                      and it will be 0 otherwise.

    """
    lenght_subarray = resolution + days_after
    start_index = random.randint(0, len(complete_series) - lenght_subarray)
    subarray = complete_series[start_index:start_index + lenght_subarray]
    if subarray[-1] >= subarray[-(days_after+1)]:
        label = 1
    else:
        label = -1
    print('valor del arreglo', subarray[-(days_after+1)], 'valor del quinto día', subarray[-1], 'etiqueta', label)
    return subarray[:-days_after], label


def save_in_pickle(identifier, subarrays, labels):
    file_pickle = f'subarrays_{identifier}.pkl'
    labels_pickle = f'labels_{identifier}.pkl'
    with open(file_pickle, 'wb') as file:
        pickle.dump(subarrays, file)
    with open(labels_pickle, 'wb') as file:
        pickle.dump(labels, file)
    return f"Data saved of the identifier {identifier}"


def get_examples(dicc_y, dicc_x_reduced, complete_series, identifier, x_discreto, x_continuo, number_of_examples):
    array_list = []  # list of the coefficient arrays obtained after running mp
    labels = []
    counter = 0
    for _ in range(number_of_examples):
        subarray, label = labeler(complete_series, 64, 5)
        labels.append(label)
        subarray_continuo = np.interp(x_continuo, x_discreto, subarray)
        r, ap, subarray_coefficients = mp(dicc_y, dicc_x_reduced, subarray_continuo, x_continuo, 100)
        counter += 1
        print(counter)
        array_list.append(subarray_coefficients)
    save_in_pickle(identifier, array_list, labels)
    return print(f"Finished of running the examples with the identifier {identifier}")


def get_examples_generalized(dicc_y, dicc_x_reduced, complete_series, identifier, x_discreto, x_continuo,
                             number_of_examples):
    array_list = []  # list of the coefficient arrays obtained after running mp
    labels = []
    counter = 0
    for _ in range(number_of_examples):
        subarray, label = labeler(complete_series, 64, 5)
        labels.append(label)
        subarray_continuo = np.interp(x_continuo, x_discreto, subarray)
        r, ap, subarray_coefficients = mp_generalized(dicc_y, dicc_x_reduced, subarray_continuo, x_continuo, 100)
        counter += 1
        print(counter)
        array_list.append(subarray_coefficients)
    save_in_pickle(identifier, array_list, labels)
    return print(f"Finished of running the examples with the identifier {identifier}")


def main(wavelet, resolution, level):
    dicc_y = gen_dict(wavelet, resolution, level)
    dicc_x = gen_dict_x(wavelet, resolution, level)

    selected_columns = [0, 64, 128, 192, 256]
    dicc_x_reduced = dicc_x[:, selected_columns]
    del dicc_x

    delta = int((dicc_x_reduced.shape[0] - 1) / 3)
    total_points = int((63 * delta) + 1)

    ### LOAD DATA FROM MACHINE
    file = 'AAPL.csv'
    complete_series = np.genfromtxt(file, delimiter=',', usecols=5)[1:]
    # datos_adj = complete_series[:resolution]

    # ### LOAD DATA
    # data = read_ts_from_ibdb('AAPL', '1 day', None, '2023-08-31', last=1000)
    # data_adj_close = data[0]['adj_close'][:resolution].to_numpy()

    ### Domain interpolation for the given delta
    x_discreto = np.arange(resolution)
    x_continuo = dicc_x_reduced[:, 0]
    for k in range(1, 21):
        aux_x = dicc_x_reduced[:, 0][1:] + 3*k
        x_continuo = np.concatenate((x_continuo, aux_x))
    del aux_x

    #####EXMAPLES SIGNALS TO THE MP#######
    # y_continuo = np.interp(x_continuo, x_discreto, data_adj_close)
    # datos_adj_continuo = np.interp(x_continuo, x_discreto, datos_adj)  #data of machine

    ### SIGNAL PLOTS
    # plt.scatter(np.arange(64), datos_adj, label='discrete')
    # plt.legend()
    # plt.show()
    # plt.plot(x_continuo, datos_adj_continuo, label='Signal')
    # plt.legend()
    # plt.show()

    ####################################################################################################
    # ##### GET EXAMPLES ##################
    # identifier = sys.argv[1]
    # get_examples(dicc_y, dicc_x_reduced, complete_series, identifier, x_discreto, x_continuo, 1)

    # ####### SIGNAL PLOTS ################
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

    # ########### EXPERIMENTO ATOMS ALETORIOS #########
    # atoms, j_s, a_s = random_atoms(dicc_y, dicc_x_reduced, x_continuo)
    # ## INTERNAL PRODUCTS
    # for i in range(5):
    #     for k in range(5):
    #         print('p_interno', np.dot(atoms[i], atoms[k])/1024, 'atomo1', j_s[i], a_s[i],
    #               'atomo2',j_s[k], a_s[k])
    # rango = range(1, 101)
    # numeros_aleatorios = random.sample(rango, 5)
    # numeros_aleatorios.sort(reverse=True)
    # print('números_aleatorios', numeros_aleatorios)
    # signal = np.zeros(64513)
    # for i in range(5):
    #     signal += numeros_aleatorios[i]*atoms[i]
    #
    # ## MATHING PURSUIT FOR THE RANDOM EXPERIMENT
    # r, ap, coefficients = mp(dicc_y, dicc_x_reduced, signal, x_continuo, 10)
    # print('indices diferentes de cero', np.nonzero(coefficients))

    # ########### EXPERIMENTO ATOMS ALETORIOS GENERALIZADO #########
    # atoms, j_s, a_s = random_atoms_generalized(dicc_y, dicc_x_reduced, x_continuo)
    # ## INTERNAL PRODUCTS
    # for i in range(5):
    #     for k in range(5):
    #         print('p_interno', np.dot(atoms[i], atoms[k])/delta, 'atomo1', j_s[i], a_s[i],
    #               'atomo2',j_s[k], a_s[k])
    # rango = range(1, 101)
    # numeros_aleatorios = random.sample(rango, 5)
    # numeros_aleatorios.sort(reverse=True)
    # print('números_aleatorios', numeros_aleatorios)
    # signal = np.zeros(total_points)
    # for i in range(5):
    #     signal += numeros_aleatorios[i]*atoms[i]
    #
    # ## MATHING PURSUIT FOR THE RANDOM EXPERIMENT GENERALIZED #####
    # r, ap, coefficients = mp_generalized(dicc_y, dicc_x_reduced, signal, x_continuo, 10)
    # print('indices diferentes de cero', np.nonzero(coefficients))

    # ##### GET EXAMPLES GENERALIZED##################
    identifier = sys.argv[1]
    get_examples_generalized(dicc_y, dicc_x_reduced, complete_series, identifier, x_discreto, x_continuo, 1)

    ### Calculating and storing the array required for executing the SVD
    # array_list = []   # list of the coefficient arrays obtained after running mp
    # labels = []
    # counter = 0
    # pid = psutil.Process().pid
    # identifier = sys.argv[1]
    # identifier = 0
    # get_examples(dicc_y, dicc_x, complete_series, identifier, x_discreto, x_continuo, 1)
    # os.sched_setaffinity(0, [identifier])
    # psutil.Process(pid).cpu_affinity([identifier])
    # identifier = int(sys.argv[1])
    # affinity.set_process_affinity([identifier])

    #############IDENTIFIERS
    # array_list = []  # list of the coefficient arrays obtained after running mp
    # labels = []
    # counter = 0
    # identifier = 0
    # for l in range(1):
    #     subarray, label = labeler(complete_series, 64)
    #     labels.append(label)
    #     subarray_continuo = np.interp(x_continuo, x_discreto, subarray)
    #     r, ap, subarray_coefficients = mp(dicc_y, dicc_x, subarray_continuo, x_continuo, 100)
    #     counter += 1
    #     print(counter)
    #     array_list.append(subarray_coefficients)
    # save_in_pickle(identifier, array_list, labels)

    # number_of_threads = int(sys.argv[1])
    # identifier = 0
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    # results = executor.map(my_function, range(2))
    # num_processes = 3
    # processes = []
    # for i in range(num_processes):
    #     my_function = get_examples(dicc_y, dicc_x, complete_series, identifier, x_discreto, x_continuo, 1)
    #     process = multiprocessing.Process(target=my_function, args=(i + 1,))
    #     processes.append(process)
    #     process.start()
    #
    # for process in processes:
    #     process.join()
    #
    # print("All processes finished.")

    # pkl_file = 'matrix_of_arrays.pkl'
    # with open(pkl_file, 'wb') as file:
    #     pickle.dump(array_list, file)

    # array_to_svd = np.column_stack(array_list)
    # print('dimensiones del arreglo', array_to_svd.shape)

    # ### MATCHING PURSUIT
    # r200, ap200, coefficients200 = mp(dicc_y, dicc_x, datos_adj_continuo, x_continuo, 200)
    # plt.plot(x_continuo, datos_adj_continuo, 'red', label='SIGNAL')
    # plt.legend()
    # # plt.show()

    # plt.plot(x_continuo, ap200, 'blue', label='APPROXIMATION')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main('db2', 64, 8)

    # data = read_ts_from_ibdb('AAPL', '1 day', None, '2023-08-31', last=1000)
    # data_adj_close = data[0]['adj_close'][:64].to_numpy()
    # print(data_adj_close)
    # print(type(data_adj_close))
    # print(len(data_adj_close))


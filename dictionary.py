from paquete import w_jnk
import numpy as np
import time
import matplotlib.pyplot as plt

def const_dict(wavelet,j, n_row):
    """
    Calculates the dictionary matrix, which contains in each column
    the coefficients of the wavelet dictionary atoms.

    Args:
        wavelet: type of wavelet
        j: scale parameter
        n_row: number of coefficients of each atom

    Returns:
        dictionary matrix
    """

    m = (2**j -1)*(n_row)
    matriz = np.zeros((n_row, m))
    for k in range(n_row):
        for n in range(2**j -1):
            arreglox = w_jnk(wavelet, j, n, k).x
            # norm = np.trapz((w_jnk(wavelet, j, n, k).y)*(w_jnk(wavelet, j, n, k).y), w_jnk(wavelet, j, n, k).x)
            # print('norm', norm)
            numeros_enteros = arreglox[arreglox.astype(int) == arreglox]
            indices_no_cero = np.nonzero(numeros_enteros)
            y = w_jnk(wavelet, j, n, k).y
            atomcolumn = np.zeros(len(y))
            atomcolumn[indices_no_cero] = y[indices_no_cero]
            atomcolumn = atomcolumn[: n_row]
            matriz = np.column_stack([matriz, atomcolumn])
            #arreglo_con = np.concatenate([numeros_enteros, np.zeros(n_row -len(numeros_enteros))])
            #matriz = np.column_stack((matriz, arreglo_con))
            #matriz = np.column_stack([matriz, numeros_enteros])
            # break
        # break
    return matriz

def const_dict_continue(wavelet, n_row):
    j = 5
    # m = ((2 ** j) - 1) * (n_row)
    columnas = []
    # matriz = np.array([])
    for j in range(j):
        for n in range(64):
            columna = w_jnk(wavelet, j, n, 0).y
            columnas.append(columna)
            matriz = np.column_stack(columnas)
            print('j', j, 'n', n, 'k', 0)
            norm = np.trapz((w_jnk(wavelet, j, n, 0).y)*(w_jnk(wavelet, j, n, 0).y), w_jnk(wavelet, j, n, 0).x)
            print('norm', norm)
            # print(len(w_jnk(wavelet, j, n, k).y))
            # print(matriz.shape)
            # print('j',j,'n',n, 'k', k)
            # arreglo_con = np.concatenate([numeros_enteros, np.zeros(n_row -len(numeros_enteros))])
            # matriz = np.column_stack((matriz, arreglo_con))
            # matriz = np.column_stack([matriz, numeros_enteros])
            # break
        # break
    return matriz

def const_dict_continue_x(wavelet, n_row):
    j = 5
    # m = ((2 ** j) - 1) * (n_row)
    columnas = []
    # matriz = np.array([])
    for j in range(j):
        for n in range(64):
            columna = w_jnk(wavelet, j, n, 0).x
            columnas.append(columna)
            matriz = np.column_stack(columnas)
            print('j', j, 'n', n, 'k', 0)
    return matriz

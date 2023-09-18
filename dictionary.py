from paquete import w_jnk
import numpy as np

def dict(wavelet,j, n_row):
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
    return matriz

matriz = dict('haar',10, 32)
print(matriz)
print(np.count_nonzero(matriz))

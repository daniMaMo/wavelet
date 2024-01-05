import pylab
import pywt
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

class Function:

    """
        This class represents a function.

        This class has the following attributes:
        - x (list): The domain.
        - y (list): The range.

        Available methods:
        - vcomp(self, factor): Makes a vertical compression.
        - plot(self, name): Draws the function.
        - transform(self, k): Makes two transformations: the first is a k horizontal translation (to the right) and
                              the second is a horizontal compression by a factor of 2.
        - atom_transform(self, j, k): Makes two transformations: the first is a k horizontal translation (to the right)
                                    and the second is a horizontal dilation by a factor of 2**j.
        - convolution(self, other_function): Computes the convolution between two functions.
        - __add__(self, other_function): Computes the sum between two functions.
        """

    def __init__(self, x, y):

        """
         This class represents a function.

        Args:
            x: The domain.
            y: The range.
        """
        self.x = x
        self.y = y

    def vcomp(self, factor):
        """
        Makes a vertical compression.
        Args:
            factor: factor of compression.

        Returns: compressed function.

        """
        rango = factor*self.y
        return Function(self.x, rango)

    def plot(self, name):
        """
        Draws the function.
        Args:
            name: name of the function.

        Returns: plot of the function with its name.

        """
        pylab.plot(self.x, self.y)
        pylab.title(name)
        return pylab.show()

    def transform(self, k):
        """
        Makes two transformations: the first is a k horizontal translation (to the right) and
        the second is a horizontal compression by a factor of 2.

        Args:
            k: factor of translation.

        Returns: the transformed function instance.
        """
        n = len(self.y)
        step = k * int(n / max(self.x))
        aux2 = np.concatenate([np.zeros(step), self.y, np.zeros(n)])
        aux3 = aux2[: 2*n]
        rango = aux3[::2]
        return Function(self.x, rango)

    def atom_transform(self, j, k):
        """
        Makes two transformations: the first is a k horizontal translation (to the right)
                                    and the second is a horizontal dilation by a factor of 2**j.
        Args:
            j: scale parameter.
            k: position parameter.

        Returns: the atom_transformed function instance.

        """
        translate = self.x + k
        dominio = (2**j)*translate
        return Function(dominio, self.y)

    def convolution(self, other_function):
        """
        Computes the convolution between two functions.
        Args:
            other_function: A instance of function.

        Returns: The value of the convolution.

        """
        if abs(np.sum(np.convolve(self.y, other_function.y, mode='full'))) < 1e-5:
            convolution = 0
        else:
            convolution = np.sum(np.convolve(self.y, other_function.y, mode='full'))
        return convolution

    def __add__(self, other_function):
        """
        Computes the sum between two functions.

        Args:
            other_function: A instance of function with same domain of given function.

        Returns: the sum instance.

        """
        rango = self.y + other_function.y
        return Function(self.x, rango)


def package(wavelet, n):
    """
    Computes the n-th package of the given wavelet.
    Args:
        wavelet: It's a wavelet include in the PyWavelets package, the wavelet most used are:
                db, sym, coif, haar, bior, rbio, dmey, cdf and shan.
        n: Frequency parameter.

    Returns: the n-th package function.

    """

    phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=10)
    h = np.sqrt(2) * np.array(pywt.Wavelet(wavelet).dec_lo)[::-1]
    g = np.sqrt(2) * np.array(pywt.Wavelet(wavelet).dec_hi)[::-1]
    l = len(h)

    if n == 0:
        return Function(x, phi)
    if n == 1:
        return Function(x, psi)
    if (n % 2) == 0 and (n != 0):
        w = package(wavelet, n/2).transform(0).vcomp(h[0])
        for k in range(1, l):
            w = w + package(wavelet, n/2).transform(k).vcomp(h[k])
        return w
    if (n % 2) == 1 and (n != 1):
        w = package(wavelet, (n-1) / 2).transform(0).vcomp(g[0])
        for k in range(1, l):
            w = w + package(wavelet, (n-1)/2).transform(k).vcomp(g[k])
        return w

def w_jnk(wavelet, j, n, k):
    """
    Computes the atoms of wavelet packets of the given wavelet.

    Args:
        wavelet: It's a wavelet include in the PyWavelets package.
        j: scale parameter.
        n: frequency parameter.
        k: position parameter.

    Returns: the atoms w_jnk.

    """
    file_path = f'/home/daniela/PycharmProjects/tesis/package{wavelet}'
    if os.path.exists(file_path):
        # print(f'The file "{file_path}" exists in the system.')
        with open(file_path, 'rb') as file:
            pack = pickle.load(file)
            y = n+64
            fun = Function(pack[:, n], pack[:, y])
            return fun.atom_transform(j, k).vcomp(2**(-j/2))
    # return package(wavelet, n).atom_transform(j, k).vcomp(2**(-j/2))

def package_load(wavelet, resolution):
    columnas_y = []
    columnas_x = []
    file_path = f'/home/daniela/PycharmProjects/tesis/package{wavelet}'
    if os.path.exists(file_path):
        print(f'The file "{file_path}" exists in the system.')
        with open(file_path, 'rb') as file:
            pack = pickle.load(file)
            # dict_ = file.read()
            return pack

    else:
        print(f'The file "{file_path}" does not exist in the system.')
        for n in range(resolution):
            columna_y = package(wavelet, n).y
            columnas_y.append(columna_y)
            pack_y = np.column_stack(columnas_y)
            columna_x = package(wavelet, n).x
            columnas_x.append(columna_x)
            pack_x = np.column_stack(columnas_x)
            pack = np.concatenate((pack_x, pack_y), axis=1)
            print(n)
        with open(f'package{wavelet}', 'wb') as file:
            pickle.dump(pack, file)
            return pack

import pylab
import pywt
import numpy as np
#import torch.nn as nn

class function:
    # Constructor: se llama cuando se crea una nueva instancia de la clase
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def vcomp(self, factor):
        rango = factor*self.y
        return function(self.x, rango)

    def plot(self, nombre):
        pylab.plot(self.x, self.y)
        pylab.title(nombre)
        return pylab.show()

    def transform(self, k):
        n = len(self.y)
        step = k * int(n / max(self.x))
        aux2 = np.concatenate([np.zeros(step), self.y, np.zeros(n)])
        aux3 = aux2[: 2*n]
        rango = aux3[::2]
        return function(self.x, rango)

    def atom_transform(self, j, k):
        translate = self.x + k
        dominio = (2**j)*(translate)
        return function(dominio, self.y)

    def convolution(self, other_function):
        if abs(np.sum(np.convolve(self.y, other_function.y, mode='full')))< 1e-5:
            convolution = 0
        else:
            convolution = np.sum(np.convolve(self.y, other_function.y, mode='full'))
        return convolution

    def __add__(self, other_function):
        rango = self.y + other_function.y
        return function(self.x, rango)

    def __sub__(self, other_function):
        dominio = np.concatenate((self.x[:-1], other_function.x[1:]))
        rango = np.concatenate((self.y[:-1], (-1)*other_function.y[1:]))
        return function(dominio, rango)

def package(wavelet, n):
    phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=7)
    h = np.sqrt(2) * np.array(pywt.Wavelet(wavelet).dec_lo)[::-1]
    g = np.sqrt(2) * np.array(pywt.Wavelet(wavelet).dec_hi)[::-1]
    l = len(h)
    if n == 0:
        return function(x, phi)
    if n == 1:
        return function(x, psi)
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

def w_jnk(wavelet, j,n,k):
    return package(wavelet, n).atom_transform(j, k).vcomp(2**(-j/2))

#print(w_jnk('haar', 1,0,2).y)

#n = 3
#nombre = n
#for k in range(8):
#    print(package('db4',k).plot(k))

#for k in range(11):
#    print(package('db2',0).convolution(package('db2',k)))

#print(w_jnk('haar', 1,0,2).plot('ejemplos'))
#print(len(w_jnk('haar', 1,1,1).y))

import numpy as np
import matplotlib.pyplot as plt


# def mp(d, y, i):
#     """
#     Finds the "best matching" projections of multidimensional data
#     onto the span of an over-complete (i.e., redundant) dictionary D.
#
#     Args:
#         d: The dictionary (column vectors as atoms) of shape (Feature Dimensions, atoms Count)
#         y: The signal.
#         i: Iterations count.
#
#     Returns:
#         List of coefficients and indices for corresponding atoms.
#
#     Examples:
#          >>>  mp( np.array([[1, 1 / 2, -1 / np.sqrt(2)], [0, np.sqrt(3) / 2, -1 / np.sqrt(2)]]), np.array([1, 1 / 2]), 2)
#         [1.0606601, 0.25000006]
#         [2, 0]
#     """
#
#     atomsdim = d.shape[0]
#     atomscount = d.shape[1]
#
#     assert i <= atomscount
#
#     r = np.array(y, dtype=np.float32)  # residue of the signal
#     a = np.array(d, dtype=np.float32)  # residual atoms
#     s = np.zeros((atomsdim, 0), dtype=np.float32)  # selected atoms
#     si = []  # indices of selected atoms in the dictionary
#     c = []  # coeficients of slected atoms in the dictionary
#
#     for _ in range(i):
#         p = r @ a  # project the residue onto valid bases
#
#         index_max = np.argmax(np.abs(p))  # index of the atom with longest projection
#         # c.append(np.max(np.abs(p)))
#         c.append(p[index_max])
#         s = np.column_stack((s, a[:, index_max]))  # addition atom of the column aindex
#         si.append(index_max)  # store of indices
#         r = r - p[index_max]*a[:, index_max]
#         a[:, index_max] = 0  # make the selected atoms invalid
#
#     return s, c, si

def matching_pursuit_chat(signal, dictionary, max_iterations):
    coefficients = np.zeros(len(dictionary))
    residual = signal.copy()

    for iteration in range(max_iterations):
        inner_products = np.dot(dictionary, residual)
        best_atom_index = np.argmax(np.abs(inner_products))
        best_atom = dictionary[best_atom_index]
        coefficient = inner_products[best_atom_index]
        coefficients[best_atom_index] += coefficient
        residual -= coefficient * best_atom

    return best_atom, coefficients

def mp_algorithm(signal, dictionary, i):
    # Initialize the approximation and the residual signal
    approx = np.zeros(len(signal))
    res = signal - approx

    # Loop until the residual signal is below the threshold
    for _ in range(i):
    # while np.linalg.norm(res) > threshold:
        # Inner product
        p = res @ dictionary

        # Find the best match between the residual signal and the dictionary
        idx = np.argmax(np.abs(p))

        # Add the matched atom to the approximation
        approx += dictionary[:, idx] * p[idx]

        # Update the residual signal
        res -= approx

        # vector
        # print(approx + res)

        # norm of residual
        print('norm', np.linalg.norm(res))

    return approx


if __name__ == "__main__":
    I = 3
    # # D =  np.array([[1,2,3,4,5,6],[1,2,4,7,8,4],[4,5,2,1,4,2]])
    # # D = np.array([[1, 1 / 2, -1 / np.sqrt(2)], [0, np.sqrt(3) / 2, -1 / np.sqrt(2)]])
    # D = np.array([[1,0 ,0],[0,1,0]])
    # y = np.array([1, 1 / 2])
    # # y = np.array([1,2,3])
    #
    # s, c, si = mp(D, y, I)
    # c = np.array(c)
    # si = np.array(si)
    #
    # # Vamos a hacer la aproximación
    # # columnas_seleccionadas = D[:, si]
    # columnas_multiplicadas = s * c
    # aprox = columnas_multiplicadas.sum(axis=1)
    # print('\ncoeficients\n', c, '\nSIndices\n', si)
    # print('aprox', aprox)
    # best, coe = matching_pursuit_chat(y,D.T,2)
    # print('\ncoeficients_chat\n', coe, '\nSIndices_chat\n', best)
    # aprox_chat = columnas_multiplicadas.sum(axis=1)
    # print('aporx_chat', aprox_chat)
    #
    # N = 1000
    # t = np.linspace(0, 1, N)
    # x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    #
    # # Create a dictionary of atoms
    # M = 10
    # atoms = np.random.randn(M, N)
    #
    # # Apply the MP algorithm
    # approx = mp_algorithm(y, D,2)
    # print('new', approx)
    #
    # plt.plot(y, label='Original signal')
    # plt.plot(approx, label='Approximation')
    # plt.legend()
    # plt.show()


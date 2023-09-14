import numpy as np

def mp(D, y, I):
    '''
            D: The dictionary (column vectors as atoms).
               Shape: (Feature Dimensions, atoms Count)
            y: The signal to sparse encode.
            I: Iterations count.
        '''

    AtomsDim = D.shape[0]
    AtomsCount = D.shape[1]

    assert I <= AtomsCount

    r = np.array(y, dtype=np.float32)  # residue of the signal
    A = np.array(D, dtype=np.float32)  # residual atoms
    S = np.zeros((AtomsDim, 0), dtype=np.float32)  # selected atoms
    si = []  # indices of selected atoms in the dictionary

    for _ in range(I):
        p = r @ A  # project the residue onto valid bases

        index_max = np.argmax(np.abs(p))  # index of the atom with longest projection
        S = np.column_stack((S, A[:, index_max]))  # addition atom of the column aindex
        si.append(index_max)  # store of indices
        r = r - p[index_max]*A[:,index_max]
        A[:, index_max] = 0 # make the selected atoms invalid
        print(r)

    return S, si

if __name__ == "__main__":
    I = 2
    D = np.array([[1, 1 / 2, -1 / np.sqrt(2)], [0, np.sqrt(3) / 2, -1 / np.sqrt(2)]])
    y = np.array([1, 1 / 2])

    S, si = mp(D, y, I)
    print('\nSIndices\n', si,  '\nS\n', S)

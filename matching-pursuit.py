import numpy as np

def mp(D, y, I):
    """
    Finds the "best matching" projections of multidimensional data
    onto the span of an over-complete (i.e., redundant) dictionary D.

    Args:
        D: The dictionary (column vectors as atoms) of shape (Feature Dimensions, atoms Count)
        y: The signal.
        I: Iterations count.

    Returns:
        List of coefficients and indices for corresponding atoms.

    Examples:
         >>>  mp( np.array([[1, 1 / 2, -1 / np.sqrt(2)], [0, np.sqrt(3) / 2, -1 / np.sqrt(2)]]), np.array([1, 1 / 2]), 2)
        [1.0606601, 0.25000006]
        [2, 0]
    """

    AtomsDim = D.shape[0]
    AtomsCount = D.shape[1]

    assert I <= AtomsCount

    r = np.array(y, dtype=np.float32)  # residue of the signal
    A = np.array(D, dtype=np.float32)  # residual atoms
    S = np.zeros((AtomsDim, 0), dtype=np.float32)  # selected atoms
    si = []  # indices of selected atoms in the dictionary
    c = [] # coeficients of slected atoms in the dictionary

    for _ in range(I):
        p = r @ A  # project the residue onto valid bases

        index_max = np.argmax(np.abs(p))  # index of the atom with longest projection
        c.append(np.max(np.abs(p)))
        S = np.column_stack((S, A[:, index_max]))  # addition atom of the column aindex
        si.append(index_max)  # store of indices
        r = r - p[index_max]*A[:,index_max]
        A[:, index_max] = 0 # make the selected atoms invalid


    return c, si

if __name__ == "__main__":
    I = 2
    D = np.array([[1, 1 / 2, -1 / np.sqrt(2)], [0, np.sqrt(3) / 2, -1 / np.sqrt(2)]])
    y = np.array([1, 1 / 2])

    c, si = mp(D, y, I)
    print('\ncoeficients\n', c, '\nSIndices\n', si)

import numpy as np
from sklearn.decomposition import TruncatedSVD

def redSVD(array, dim):
    """
    Reduces dimension of the given array (data).

    Args:
        array: array of data.
        dim: desired dimension (columns number of the data).

    Returns:
        The new array of shape: (same row number of "array", "dim" )

    Examples:
        redSVD(np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]], 2)
        [[1, 2], [4, 5], [7, 8]]
    """

    assert dim <= array.shape[1]

    tsvd = TruncatedSVD(n_components=dim)    # Create the instance TruncatedSVD
    array_aprox = tsvd.fit_transform(array)  # It's only U @ Sigma

    return np.dot(array_aprox, tsvd.components_[:,:dim])



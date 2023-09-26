import numpy as np
from sklearn.decomposition import TruncatedSVD


def cenandstand(array):
    """
    This function centers and standardizes the data entered.

    Args:
        array: the data entered

    Returns:
        An array of data centered and standardized.

    Example:
        cenandstand(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [[-1.22474487 -1.22474487 -1.22474487], [0. 0. 0.], [1.22474487 1.22474487 1.22474487]]
    """

    # Center the data (subtract the mean of each column)
    mean_columns = np.mean(array, axis=0)  # calculates the average per column.
    data_central = array - mean_columns

    # Standardize the data (divide by the standard deviation of each column)
    deviation_column = np.std(array, axis=0)  # calculates the standard deviation per column.
    data_standardized = data_central / deviation_column

    return data_standardized


def redSVD(array, dim):
    """
    WARNING: Data must be centered and standardized.

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

    return np.dot(array_aprox, tsvd.components_[:, :dim])

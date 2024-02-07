import numpy as np
import matplotlib.pyplot as plt
import random


def interpolation(dicc_y, dicc_x_reduced, j, a, k):
    index_start = k * 1024
    support = np.linspace(k, (2 ** j) * 3 + k, num=((3072 * (2 ** j)) + 1))
    atom_scala = np.interp(support, dicc_x_reduced[:, j] + k, dicc_y[:, a])

    if k in range(64 - 3 * (2 ** j)):
        atom_complete = np.pad(atom_scala, (index_start, 64513 - index_start - len(atom_scala)), mode='constant')
    elif k in range(-3 * (2 ** j) + 1, 0):
        atom_complete = np.pad(atom_scala, (0, 64513 - index_start - len(atom_scala)), mode='constant')[np.abs(k) * 1024:]
    elif k in range(64 - 3 * (2 ** j), 63):
        atom_complete = np.pad(atom_scala, (index_start, 0), mode='constant')[:64513]

    return atom_complete

def interpolation_generalized(dicc_y, dicc_x_reduced, j, a, k):
    len_atom = dicc_x_reduced.shape[0] - 1   # length of atoms without interpolation
    delta = int((dicc_x_reduced.shape[0] - 1)/3)
    total_points = int((63*delta) + 1)
    index_start = k * delta
    support = np.linspace(k, (2 ** j) * 3 + k, num=((len_atom * (2 ** j)) + 1))
    atom_scala = np.interp(support, dicc_x_reduced[:, j] + k, dicc_y[:, a])

    if k in range(64 - 3 * (2 ** j)):
        atom_complete = np.pad(atom_scala, (index_start, total_points - index_start - len(atom_scala)), mode='constant')
    elif k in range(-3 * (2 ** j) + 1, 0):
        atom_complete = np.pad(atom_scala, (0, total_points - index_start - len(atom_scala)), mode='constant')[
                        np.abs(k) * delta:]
    elif k in range(64 - 3 * (2 ** j), 63):
        atom_complete = np.pad(atom_scala, (index_start, 0), mode='constant')[:total_points]

    return atom_complete

def random_atoms(dicc_y, dicc_x_reduced, x_continuo):
    """
    Generates a list of 5 random atoms based on the given dictionaries.

    Args:
    - dicc_y: Dictionary representing y values.
    - dicc_x: Dictionary representing x values.
    - x_continuo: Continuous domain values with their respective delta.

    Returns:
    - atoms: List of 5 randomly generated continuous atoms.
    - j_values: List containing 'j' (scale) values for each iteration.
    - a_values: List containing 'a' (atom number in the dictionary) values for each iteration.
    """

    atoms = []
    j_values = []
    a_values = []

    random.seed(90)
    for i in range(5):
        a = random.choice(range(320))
        a_values.append(a)
        j = a // 64
        j_values.append(j)
        k = random.choice(range(-3 * (2 ** j) + 1, 63))
        print(j, a, k)

        atoms.append(interpolation(dicc_y, dicc_x_reduced, j, a, k))

        plt.plot(x_continuo, atoms[i], color='red', label=f'{i}')
        plt.legend()
        plt.show()

    return atoms, j_values, a_values

def random_atoms_generalized(dicc_y, dicc_x_reduced, x_continuo):
    """
    Generates a list of 5 random atoms based on the given dictionaries.

    Args:
    - dicc_y: Dictionary representing y values.
    - dicc_x: Dictionary representing x values.
    - x_continuo: Continuous domain values with their respective delta.

    Returns:
    - atoms: List of 5 randomly generated continuous atoms.
    - j_values: List containing 'j' (scale) values for each iteration.
    - a_values: List containing 'a' (atom number in the dictionary) values for each iteration.
    """

    atoms = []
    j_values = []
    a_values = []

    random.seed(90)
    for i in range(5):
        a = random.choice(range(320))
        a_values.append(a)
        j = a // 64
        j_values.append(j)
        k = random.choice(range(-3 * (2 ** j) + 1, 63))
        print(j, a, k)

        atoms.append(interpolation_generalized(dicc_y, dicc_x_reduced, j, a, k))

        plt.plot(x_continuo, atoms[i], color='red', label=f'{i}')
        plt.legend()
        plt.show()

    return atoms, j_values, a_values

def check(dicc_y, dicc_x, scale):
    """
    Check if the dictionary is orthogonal at a given scale.
    Args:
        dicc_y: Dictionary representing y values.
        dicc_x: Dictionary representing x values.
        scale: The scale at which to check the dictionaries.

    Returns:
    - If the dictionary is orthogonal, it returns the expected result.
    - Otherwise, it returns the incorrect value with its corresponding atoms.

    """
    assert scale < 5
    j = scale
    for a in range(64 * j, 64 * (j + 1)):
        for b in range(64 * j, 64 * (j + 1)):
            aux = np.linspace(0, (2 ** j) * 3, num=((3072 * (2 ** j)) + 1))
            atom_scala_a = np.interp(aux, dicc_x[:, a], dicc_y[:, a])
            atom_scala_b = np.interp(aux, dicc_x[:, b], dicc_y[:, b])
            if a == b and (abs(np.dot(atom_scala_a, atom_scala_b)/1024)-1) < 0.2:
                pass
                # print('correct, unitary norm')
            elif abs(np.dot(atom_scala_a, atom_scala_b)/1024) < 0.2:
                pass
                # print('correct, different atoms are orthogonal')
            else:
                print(np.dot(atom_scala_a, atom_scala_b)/1024, a, b)
    return

def mp(dicc_y, dicc_x_reduced, y_signal, x_continuo, iteration):
    """
    Finds the "best matching" projections of multidimensional data
    onto the span of an over-complete (i.e., redundant) dictionary D.

    Args:
        dicc_y: Dictionary representing y values.
        dicc_x: Dictionary representing x values.
        y_signal: The signal that you want to approach.
        x_continuo: Continuous domain with the appropriate delta.
        iteration: Number of iterations.

    Returns: A list in the following order with the following information:
    Residue, the approximation after all iterations.

    """
    assert iteration <= 25792  ## To j = 5, n=64
    r = y_signal
    approx = np.zeros(64513)
    coefficients = np.zeros(25792)

    for _ in range(iteration):
        p = {}
        key = 0
        print(_)

        for a in range(320):
            j = a // 64

            for k in range(-3 * (2 ** j)+1, 63):
                atom_comp = interpolation(dicc_y, dicc_x_reduced, j, a, k)
                new_value = {'product': (atom_comp @ r) * (1 / 1024), 'list': [j, a, k]}
                p[key] = new_value
                key += 1

        index_c_max = max(p, key=lambda k: p[k]['product'])
        c = p[index_c_max]['product']
        # print('c', c)
        # print('el valor maximo de las keys', max(p.keys()))
        coefficients[index_c_max] = c

        ### Calculating the residue
        list_max = p[index_c_max]['list']  # list: j, a , k
        # print(list_max)
        k_max = list_max[2]
        j_max = list_max[0]
        a = list_max[1]

        a_max = interpolation(dicc_y, dicc_x_reduced, j_max, a, k_max)

        ### Plot signal
        # plt.plot(x_continuo, r, label=f'signal{_}')
        # plt.legend()

        ### RESIDUE
        r = r - c * a_max

        # plt.plot(x_continuo, c * a_max, label='atom_max')
        # plt.legend()
        #
        # plt.plot(x_continuo, r, label='residuo')
        # plt.legend()
        # plt.show()

        ## APPROXIMATION
        approx = approx + (c * a_max)
        # plt.plot(x_continuo, y_signal, label='Signal')
        # plt.legend()
        # plt.plot(x_continuo, approx, label=f'Approx{_}')
        # plt.legend()
        # plt.show()

    return r, approx, coefficients

def mp_generalized(dicc_y, dicc_x_reduced, y_signal, x_continuo, iteration):
    """
    Finds the "best matching" projections of multidimensional data
    onto the span of an over-complete (i.e., redundant) dictionary D.

    Args:
        dicc_y: Dictionary representing y values.
        dicc_x: Dictionary representing x values.
        y_signal: The signal that you want to approach.
        x_continuo: Continuous domain with the appropriate delta.
        iteration: Number of iterations.

    Returns: A list in the following order with the following information:
    Residue, the approximation after all iterations.

    """
    assert iteration <= 25792  ## To j = 5, n=64
    len_atom = dicc_x_reduced.shape[0] - 1  # length of atoms without interpolation
    delta = int((dicc_x_reduced.shape[0] - 1) / 3)
    total_points = int((63 * delta) + 1)
    r = y_signal
    approx = np.zeros(total_points)
    coefficients = np.zeros(25792)

    for _ in range(iteration):
        p = {}
        key = 0
        print(_)

        for a in range(320):
            j = a // 64

            for k in range(-3 * (2 ** j)+1, 63):
                atom_comp = interpolation_generalized(dicc_y, dicc_x_reduced, j, a, k)
                new_value = {'product': (atom_comp @ r) * (1 / delta), 'list': [j, a, k]}
                p[key] = new_value
                key += 1

        index_c_max = max(p, key=lambda k: p[k]['product'])
        c = p[index_c_max]['product']
        # print('c', c)
        # print('el valor maximo de las keys', max(p.keys()))
        coefficients[index_c_max] = c

        ### Calculating the residue
        list_max = p[index_c_max]['list']  # list: j, a , k
        # print(list_max)
        k_max = list_max[2]
        j_max = list_max[0]
        a = list_max[1]

        a_max = interpolation_generalized(dicc_y, dicc_x_reduced, j_max, a, k_max)

        ### Plot signal
        # plt.plot(x_continuo, r, label=f'signal{_}')
        # plt.legend()

        ### RESIDUE
        r = r - c * a_max

        # plt.plot(x_continuo, c * a_max, label='atom_max')
        # plt.legend()
        #
        # plt.plot(x_continuo, r, label='residuo')
        # plt.legend()
        # plt.show()

        ## APPROXIMATION
        approx = approx + (c * a_max)
        # plt.plot(x_continuo, y_signal, label='Signal')
        # plt.legend()
        # plt.plot(x_continuo, approx, label=f'Approx{_}')
        # plt.legend()
        # plt.show()

    return r, approx, coefficients
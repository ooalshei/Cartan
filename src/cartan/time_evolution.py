# Python version 3.11.5
# Modified on December 20, 2023

################ Imports ################
import numpy as np
import scipy.linalg as la
from algebras import InvolutionlessCartan

#########################################

STRING_TO_MATRIX = {'I': [[1, 0], [0, 1]], 'X': [[0, 1], [1, 0]], 'Y': [[0, -1j], [1j, 0]], 'Z': [[1, 0], [0, -1]],
                    0: [[1, 0], [0, 1]], 1: [[0, 1], [1, 0]], 2: [[0, -1j], [1j, 0]], 3: [[1, 0], [0, -1]]}


def singles_product(angles, strings):
    N = len(strings[0])
    unitary = np.identity(2 ** N)
    identity = np.identity(2 ** N)

    for i in range(len(strings)):
        string_matrix = STRING_TO_MATRIX[strings[i][0]]
        for j in range(len(strings[i]) - 1):
            string_matrix = np.kron(string_matrix, STRING_TO_MATRIX[strings[i][j + 1]])
        unitary = unitary @ (np.cos(angles[i]) * identity + 1j * np.sin(angles[i]) * string_matrix)
    return unitary


def exact_unitary(hamiltonian_dict, simulation_time):
    unitary_generator = np.zeros(2 ** len(next(iter(hamiltonian_dict))))
    for hamiltonian_string in hamiltonian_dict.keys():
        string_matrix = STRING_TO_MATRIX[hamiltonian_string[0]]
        for j in range(len(hamiltonian_string) - 1):
            string_matrix = np.kron(string_matrix, STRING_TO_MATRIX[hamiltonian_string[j + 1]])
        unitary_generator = unitary_generator + hamiltonian_dict[hamiltonian_string] * string_matrix
    return la.expm(-1j * simulation_time * unitary_generator)


def trotter_unitary(number_of_sites, model, parameters, simulation_time, steps):
    time_step = simulation_time / steps
    hamiltonian_strings, coefficient_list = InvolutionlessCartan(number_of_sites, model).builder(parameters)
    theta = -time_step * np.array(coefficient_list)
    unitary = singles_product(theta, hamiltonian_strings)

    onestep = unitary.copy()
    for i in range(steps - 1):
        unitary = unitary @ onestep

    return unitary, onestep


def cartan_unitary(angles, generators, subalgebra_dict, simulation_time):
    unitary = singles_product(angles, generators)
    h_angles = -simulation_time * np.array(list(subalgebra_dict.values()))
    h_strings = list(subalgebra_dict.keys())
    unitary = unitary @ singles_product(h_angles, h_strings)
    reversed_angles = -np.array(angles[::-1])
    reversed_generators = generators[::-1]
    return unitary @ singles_product(reversed_angles, reversed_generators)

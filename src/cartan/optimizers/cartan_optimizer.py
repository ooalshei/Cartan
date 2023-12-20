# Python version 3.11.5
# Modified on December 20, 2023


########### Imports ###########
import numpy as np
import scipy.optimize as opt
from .. import pauli_operations

###############################


def mut_irr(n: int, x: float = np.pi) -> list[float]:
    """Returns a list of n mutually irrational numbers. Takes the optional argument x irrational to build the set"""
    y = x % 1
    numbers = [y]
    for i in range(1, n):
        y = (x * y) % 1
        numbers.append(y)

    return numbers


def cost_function(angles, algebra_strings, subalgebra_element, hamiltonian_dict):
    sentence1 = pauli_operations.exp_conjugation(algebra_strings, angles, subalgebra_element)
    sentence2 = pauli_operations.full_product(sentence1, hamiltonian_dict)
    return np.real(pauli_operations.trace(sentence2))


def optimizer(hamiltonian_dict, algebra_strings, subalgebra_strings, initial_angles=None, method="BFGS",
              tol=1e-6, coefficient_tol=1e-6):
    angles_initial = np.pi * np.random.rand(len(algebra_strings)) if initial_angles is None else initial_angles.copy()
    numbers = mut_irr(len(subalgebra_strings))
    h_element = dict(zip(subalgebra_strings, numbers))

    def f(x):
        return cost_function(x, algebra_strings, h_element, hamiltonian_dict)

    angles = opt.minimize(f, angles_initial, method=method, tol=tol).x % np.pi
    reversed_angles = np.flip(-angles)
    reversed_strings = algebra_strings[::-1]
    transformed_hamiltonian_dict = pauli_operations.exp_conjugation(reversed_strings, reversed_angles, hamiltonian_dict)

    i = 0
    hamiltonian_tuples = list(transformed_hamiltonian_dict.items())
    while i < len(hamiltonian_tuples):
        if np.abs(hamiltonian_tuples[i][1]) <= coefficient_tol:
            hamiltonian_tuples.pop(i)
        else:
            i += 1
    transformed_hamiltonian_dict = dict(hamiltonian_tuples)

    return angles, algebra_strings, transformed_hamiltonian_dict

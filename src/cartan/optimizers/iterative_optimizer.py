# Python version 3.11.5
# Created on December 20, 2023

##################### Imports #####################
import numpy as np
from ..optimizers.cartan_optimizer import optimizer
from ..pauli_operations import exp_conjugation

###################################################


def iterative_optimizer(hamiltonian_dict: dict[tuple[int, ...], float],
                        subspace_strings: list[list[tuple[int, ...]]],
                        abelian_strings: list[tuple[int, ...]],
                        initial_angles: list[list[float]] | None = None,
                        method: str = "BFGS",
                        tol: float = 1e-6,
                        iterations: float = None,
                        coefficient_tol: float = 1e-6
                        ) -> tuple[list[list[float]], list[list[tuple[int, ...]]], dict[tuple[int, ...], complex],
                             dict[tuple[int, ...], complex]]:

    if initial_angles is None:
        angles = []
        for subspace in subspace_strings:
            angles.append(np.random.rand(len(subspace)))
    else:
        angles = initial_angles.copy()
    diagonal_hamiltonian = hamiltonian_dict.copy()
    transformed_hamiltonian = hamiltonian_dict.copy()

    for i in range(len(abelian_strings)):
        angles[i], subspace, diagonal_hamiltonian, _ = optimizer(diagonal_hamiltonian,
                                                                 subspace_strings[i],
                                                                 [abelian_strings[i]],
                                                                 angles[i],  # type: ignore
                                                                 method=method,
                                                                 tol=tol,
                                                                 iterations=iterations,
                                                                 coefficient_tol=coefficient_tol
                                                                 )

        reversed_angles = np.flip(-angles[i])
        reversed_subspace = subspace[::-1]
        transformed_hamiltonian = exp_conjugation(reversed_subspace, reversed_angles, transformed_hamiltonian)
    return angles, subspace_strings, diagonal_hamiltonian, transformed_hamiltonian

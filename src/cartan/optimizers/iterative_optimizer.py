import numpy as np

from ..optimizers.cartan_optimizer import optimizer
from ..pauli_operations import exp_conjugation


def iterative_optimizer(hamiltonian_dict: dict[tuple[int], float],
                        subspace_strings: list[list[tuple[int]]],
                        abelian_strings: list[tuple[int]],
                        initial_angles: list[list[float]] | None = None,
                        method: str = "BFGS",
                        tol: float = 1e-6,
                        tol_type: str = "rel",
                        iterations: float = None,
                        coefficient_tol: float = 1e-6
                        ) -> dict[str, list[float | tuple[int]] | dict[tuple[int], float]]:
    if initial_angles is None:
        angles = []
        for subspace in subspace_strings:
            angles.append(np.random.rand(len(subspace)))
    else:
        angles = initial_angles.copy()
    diagonal_hamiltonian = hamiltonian_dict.copy()
    transformed_hamiltonian = hamiltonian_dict.copy()

    calls = 0
    for i in range(len(abelian_strings)):
        result = optimizer(diagonal_hamiltonian,
                           subspace_strings[i],
                           [abelian_strings[i]],
                           angles[i],  # type: ignore
                           method=method,
                           tol=tol,
                           tol_type=tol_type,
                           iterations=iterations,
                           coefficient_tol=coefficient_tol
                           )

        angles[i] = result["angles"]
        reversed_angles = np.flip(-angles[i])
        reversed_subspace = result["k"][::-1]
        diagonal_hamiltonian = result["H_diagonal"]
        calls += result["calls"]
        relative_error = result["rel_error"]
        iteration = result["iterations"]
        transformed_hamiltonian = exp_conjugation(reversed_subspace, reversed_angles, transformed_hamiltonian)
    return {"angles": angles, "k": subspace_strings, "H_diagonal": diagonal_hamiltonian,
            "H_transformed": transformed_hamiltonian, "rel_error": relative_error,
            "iterations": iteration, "calls": calls}

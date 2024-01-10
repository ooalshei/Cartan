import numpy as np
import scipy.optimize as opt
from .. import pauli_operations


def _mut_irr(length: int, seed: float = np.pi) -> list[float]:
    r"""
    Returns a list of n mutually irrational numbers.

    Parameters:
    -----------
    length : int
        The count of irrational numbers needed.
    seed : float, optional
        The seed that is used for irrational number generation. Must be irrational. Default value is :math:`\pi`.

    Returns:
    --------
    numbers : list[float]
        The mutually irrational numbers.
    """
    y = seed % 1
    numbers = [y]
    for i in range(1, length):
        y = (seed * y) % 1
        numbers.append(y)

    return numbers


def _cost_function(angles: list[float],
                   algebra_strings: list[tuple[int, ...]],
                   subalgebra_element: dict[tuple[int, ...], float],
                   hamiltonian_dict: dict[tuple[int, ...], float],
                   tol: float = 0) -> float:
    r"""
    Calculates the cost function :math:`\mathrm{Tr}(KvK^\dag, \mathcal{H})`.

    Parameters:
    -----------
    angles : list[float]
        The angles appearing in the exponents of K.
    algebra_strings : list[tuple[int, ...]]
        The Pauli strings appearing in the exponents of K.
    subalgebra_element : dict[tuple[int, ...], float]
        The element belonging to the Cartan subalgebra.
    hamiltonian_dict : dict[tuple[int, ...], float]
        The Hamiltonian dictionary.
    tol : float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    float
        The cost function.
    """
    sentence1 = pauli_operations.exp_conjugation(algebra_strings, angles, subalgebra_element, tol)
    sentence2 = pauli_operations.full_product(sentence1, hamiltonian_dict, tol)
    return np.real(pauli_operations.trace(sentence2))


def optimizer(hamiltonian_dict: dict[tuple[int, ...], float],
              algebra_strings: list[tuple[int, ...]],
              subalgebra_strings: list[tuple[int, ...]],
              initial_angles: list[float] | None = None,
              method: str = "BFGS",
              tol: float = 1e-6,
              iterations: float = None,
              coefficient_tol: float = 0) -> dict[str, list[float | tuple[int, ...]] |
                                                       dict[tuple[int, ...], float]]:
    """
    Performs the optimization procedure.

    Parameters:
    -----------
    hamiltonian_dict : dict[tuple[int, ...], float]
        The original Hamiltonian sentence.
    algebra_strings : list[tuple[int, ...]]
        The Pauli strings appearing in the exponents of K.
    subalgebra_strings : list[tuple[int, ...]]
        Single strings that form a basis for the Cartan subalgebra.
    initial_angles : list[float], optional
        Takes a set of angles to start the iterative procedure.
    method : str, default: "BFGS"
        The optimization scheme. Must be either "roto" or a method accepted by SciPy.
    tol : float, default=1e-6
        Optimization tolerance. Non-negative number.
    iterations : int, optional
        The maximum number of iteration to perform.
    coefficient_tol : float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    angles : list[float]
        The optimum angles. Use key "angles".
    algebra_strings : list[tuple[int, ...]]
        The Pauli strings appearing in the exponents of K. Use key "k".
    diagonal_hamiltonian : dict[tuple[int, ...], float]
        The part of the transformed Hamiltonian lying in the Cartan subalgebra. Use key "H_diagonal".
    transformed_hamiltonian : dict[tuple[int, ...], float]
        The full transformed Hamiltonian. Use key "H_transformed".
    """

    angles = np.pi * np.random.rand(len(algebra_strings)) if initial_angles is None else initial_angles.copy()
    numbers = _mut_irr(len(subalgebra_strings))
    h_element = dict(zip(subalgebra_strings, numbers))

    if method == "roto":

        def harmonic(theta, amplitude, phase, constant):
            return amplitude * np.cos(2 * theta + phase) + constant

        theta_points = np.linspace(0, np.pi / 2, 3)
        iteration = 0

        while True:
            generators_to_append = []
            existing_dict = pauli_operations.exp_conjugation(algebra_strings, angles, h_element, coefficient_tol)

            for i in range(len(algebra_strings)):
                generators_to_append.append(algebra_strings[i])
                existing_dict = pauli_operations.exp_conjugation(algebra_strings[i], -angles[i], existing_dict,
                                                                 coefficient_tol)

                angles1 = angles.copy()
                angles1[i] = theta_points[0]
                angles2 = angles.copy()
                angles2[i] = theta_points[1]
                angles3 = angles.copy()
                angles3[i] = theta_points[2]

                cost_function1 = _cost_function(angles1[:i + 1], generators_to_append, existing_dict, hamiltonian_dict,
                                                coefficient_tol)
                cost_function2 = _cost_function(angles2[:i + 1], generators_to_append, existing_dict, hamiltonian_dict,
                                                coefficient_tol)
                cost_function3 = _cost_function(angles3[:i + 1], generators_to_append, existing_dict, hamiltonian_dict,
                                                coefficient_tol)

                fit_coefficients = opt.curve_fit(harmonic, theta_points,
                                                 [cost_function1, cost_function2, cost_function3])[0]
                angle_min = -fit_coefficients[1] / 2 if fit_coefficients[0] < 0 else (np.pi - fit_coefficients[1]) / 2
                angles[i] = angle_min

            iteration += 1
            reversed_angles = np.flip(-angles)
            reversed_strings = algebra_strings[::-1]
            transformed_hamiltonian = pauli_operations.exp_conjugation(reversed_strings, reversed_angles,
                                                                       hamiltonian_dict,
                                                                       coefficient_tol)

            full_norm = 0
            error_norm = 0
            diagonal_hamiltonian = transformed_hamiltonian.copy()
            for key in transformed_hamiltonian:
                c = True
                coefficient = abs(transformed_hamiltonian[key]) ** 2
                full_norm += coefficient
                for string in subalgebra_strings:
                    if not pauli_operations.string_product(key, string)[2]:
                        c = False
                if not c:
                    error_norm += coefficient
                    diagonal_hamiltonian.pop(key)

            if iteration % 50 == 0:
                print(f"Iteration {iteration}. Relative error: {np.sqrt(error_norm / full_norm)}")

            if np.sqrt(error_norm / full_norm) <= tol or iteration == iterations:
                print(f"Total iterations: {iteration}. Relative error: {np.sqrt(error_norm / full_norm)}")
                return {"angles": angles, "k": algebra_strings, "H_diagonal": diagonal_hamiltonian,
                        "H_transformed": transformed_hamiltonian}

    else:
        def f(x):
            return _cost_function(x, algebra_strings, h_element, hamiltonian_dict)

        angles = opt.minimize(f, angles, method=method, tol=tol).x % np.pi
        reversed_angles = np.flip(-angles)
        reversed_strings = algebra_strings[::-1]
        transformed_hamiltonian = pauli_operations.exp_conjugation(reversed_strings, reversed_angles,
                                                                   hamiltonian_dict)

        full_norm = 0
        error_norm = 0
        diagonal_hamiltonian = transformed_hamiltonian.copy()
        for key in transformed_hamiltonian:
            c = True
            coefficient = abs(transformed_hamiltonian[key]) ** 2
            full_norm += coefficient
            for string in subalgebra_strings:
                if not pauli_operations.string_product(key, string)[2]:
                    c = False
            if not c:
                error_norm += coefficient
                diagonal_hamiltonian.pop(key)

        print(f"Relative error: {np.sqrt(error_norm / full_norm)}")
        return {"angles": angles, "k": algebra_strings, "H_diagonal": diagonal_hamiltonian,
                "H_transformed": transformed_hamiltonian}

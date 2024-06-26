"""
Pauli Operations
----------------
This module contains several useful operations on Pauli strings. The identity along with the three Pauli matrices are
interchangeably referred to as (0, 1, 2, 3) or (-, X, Y, Z).
"""
import numpy as np

# These arrays are used to find products of Pauli matrices.
RULES = np.array([[0, 1, 2, 3],
                  [1, 0, 3, 2],
                  [2, 3, 0, 1],
                  [3, 2, 1, 0]])
SIGN_RULES = np.array([[1, 1, 1, 1],
                       [1, 1, 1j, -1j],
                       [1, -1j, 1, 1j],
                       [1, 1j, -1j, 1]])

# These arrays are used to convert between the string and tuple representations of Pauli matrices.
NUMBERS_TO_LETTERS = {0: "-", 1: "X", 2: "Y", 3: "Z"}
LETTERS_TO_NUMBERS = {"-": 0, "X": 1, "Y": 2, "Z": 3}


def product(string1: tuple[int],
            string2: tuple[int]) -> tuple[tuple[int], complex, bool]:
    """
    Returns the signed product of two Pauli strings and whether they commute.

    Parameters:
    -----------
    string1, string2 : tuple[int]
        The two Pauli strings.

    Returns:
    --------
    result : tuple[int]
        The product of the two strings.
    sign : complex
        The sign of the product.
    bool
       True if the two strings commute and False otherwise.
    """
    # Consistency check
    if len(string1) != len(string2):
        raise Exception(f"Dimension mismatch ({len(string1)} and {len(string2)})")

    string1_array = np.array(string1)
    string2_array = np.array(string2)
    sign = np.prod(SIGN_RULES[string1_array, string2_array])
    result = tuple(RULES[string1_array, string2_array])

    if sign.imag == 0:
        return result, sign, True  # type: ignore
    else:
        return result, sign, False  # type: ignore


def strings_to_dict(strings: list[tuple[int]] | tuple[int],
                    coefficients: list[complex] | complex) -> dict[tuple[int], complex]:
    """
    Returns a dictionary for a Pauli sentence with the Pauli strings as keys and their corresponding coefficients as
    values.

    Parameters:
    -----------
    strings : list[tuple[int, ...]] or tuple[int, ...]
        Can take one Pauli string or a list of Pauli strings.
    coefficients : list[complex] or complex
        Can take one coefficient or a list of coefficients.

    Returns:
    --------
    dict[tuple[int, ...], complex]
        The Pauli sentence as a dictionary.
    """
    # Data reformatting
    coefficients_array = np.array([coefficients]).flatten()
    strings_array = np.array([strings])
    if len(strings_array.shape) > 2:
        strings_array = np.squeeze(strings_array, axis=0)
    # Consistency check
    if len(strings_array) != len(coefficients_array):
        raise Exception(f"Length mismatch - strings: {len(strings_array)}, coefficients: {len(coefficients_array)}")

    strings_tuple = tuple(map(tuple, strings_array))
    return dict(zip(strings_tuple, coefficients_array))


def print_letters(sentence: dict[tuple[int], complex] = None,
                  string_list: list[tuple[int]] = None,
                  file: str | None = None) -> None:
    """
    Prints the Pauli sentence with strings represented as (-, X, Y, Z). Must specify one and only one keyword.

    Parameters:
    -----------
    sentence : dict[tuple[int, ...], complex], optional
        The Pauli sentence as a dictionary with tuples of integers representing the Pauli strings.
    string_list : list[tuple[int, ...]], optional
        A list of Pauli strings with tuples of integers representing the Pauli strings.
    file : str, optional
        The file to print the Pauli strings to.
    """
    if sentence:
        letter_dict = {}
        for key in sentence.keys():
            string = ""
            for pauli in key:
                string += NUMBERS_TO_LETTERS[pauli]
            letter_dict[string] = sentence[key]
        print(letter_dict, file=file)

    elif string_list:
        letter_list = []
        for item in string_list:
            string = ""
            for pauli in item:
                string += NUMBERS_TO_LETTERS[pauli]
            letter_list.append(string)
        print(letter_list, file=file)


def full_sum(sentence1: dict[tuple[int], complex],
             sentence2: dict[tuple[int], complex],
             tol: float = 0) -> dict[tuple[int], complex]:
    """
    Finds the sum of two Pauli sentences.

    Parameters:
    -----------
    sentence1, sentence2 : dict[tuple[int, ...], complex]
        The two Pauli sentences.
    tol : float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    result : dict[tuple[int, ...], complex]
        The sum of the two Pauli sentences as a dictionary.
    """
    result = sentence1.copy()
    for key in sentence2.keys():
        result[key] = result.get(key, 0) + sentence2[key]
        if abs(result[key]) <= tol:
            result.pop(key)

    return result


def full_product(sentence1: dict[tuple[int], complex],
                 sentence2: dict[tuple[int], complex],
                 tol: float = 0) -> dict[tuple[int, ...], complex]:
    """
    Finds the product of two Pauli sentences.

    Parameters:
    -----------
    sentence1, sentence2 : dict[tuple[int, ...], complex]
        The two Pauli sentences.
    tol : float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    result : dict[tuple[int, ...], complex]
        The product of the two Pauli sentences as a dictionary.
    """
    result: dict[tuple[int, ...], complex] = {}
    for key1 in sentence1.keys():
        for key2 in sentence2.keys():
            string, sign, c = product(key1, key2)
            result[string] = result.get(
                string, 0) + sign * sentence1[key1] * sentence2[key2]
            if abs(result[string]) <= tol:
                result.pop(string)

    return result


def string_exp(string: tuple[int],
               angle: float) -> dict[tuple[int], complex]:
    r"""
    Finds the exponential of a Pauli string :math:`\mathrm{e}^{\mathrm{i} x P} = \cos{x} + \mathrm{i}P\sin{x}`.

    Parameters:
    -----------
    string : tuple[int, ...]
        The Pauli string to be exponentiated.
    angle : float
        The angle of rotation.

    Returns:
    --------
    result : dict[tuple[int, ...], complex]
        The resulting Pauli sentence as a dictionary.
    """
    result = {}
    if np.cos(angle) != 0:
        result[(0,) * len(string)] = np.cos(angle)
    if np.sin(angle) != 0:
        result[string] = 1j * np.sin(angle)
    return result


def exp_conjugation(generators: list[tuple[int]] | tuple[int],
                    angles: list[float] | float,
                    sentence: dict[tuple[int], complex],
                    tol: float = 0) -> dict[tuple[int], float | complex]:
    r"""
    Returns the conjugation of a Pauli sentence :math:`\mathrm{e}^{\mathrm{i} x_{1} P_1} ...
    \mathrm{e}^{\mathrm{i} x_n P_n} X \mathrm{e}^{-\mathrm{i} x_{n} P_n} ... \mathrm{e}^{-\mathrm{i} x_1 P_1}`.
    
    Parameters:
    -----------
    generators : list[tuple[int, ...]] or tuple[int, ...]
        Can take a one Pauli string or a list of Pauli strings to be exponentiated.
    angles : list[float] or float
        Can take one angle or a list of angles.
    sentence : dict[tuple[int, ...], complex]
        The Pauli sentence to be conjugated.
    tol : float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    result : dict[tuple[int, ...], complex]
        The resulting Pauli sentence as a dictionary.
    """
    # Data reformatting
    angles_array = np.array([angles]).flatten()
    cosine_array = np.cos(2 * angles_array)
    sine_array = 1j * np.sin(2 * angles_array)
    generators_array = np.array([generators])
    if len(generators_array.shape) > 2:
        generators_array = np.squeeze(generators_array, axis=0)

    # Consistency check
    if len(generators_array) != len(angles_array):
        raise Exception(f"Length mismatch - generators: {len(generators_array)}, angles: {len(angles_array)}")

    result = sentence.copy()
    for i in range(len(angles_array) - 1, -1, -1):
        temp: dict[tuple[int, ...], complex] = {}
        for key in result:
            coefficient = result[key]
            string, sign, c = product(tuple(generators_array[i]), key)  # type: ignore
            # If the ith exponent commutes with string (key) in the Pauli sentence do nothing
            if c:
                temp[key] = temp.get(key, 0) + coefficient
            # If it doesn't commute it necessary anticommutes. Perform the operation exp(2ixP).string
            else:
                temp[key] = temp.get(key, 0) + cosine_array[i] * coefficient
                temp[string] = temp.get(string, 0) + sign * sine_array[i] * coefficient

                if abs(temp[string]) <= tol:
                    temp.pop(string)
            if abs(temp[key]) <= tol:
                temp.pop(key)
        result = temp.copy()
    return result


def trace(sentence: dict[tuple[int, ...], complex]) -> float | complex:
    """
    Finds the normalized trace of a Pauli sentence.
    
    Parameters:
    -----------
    sentence : dict[tuple[int, ...], complex]
        The Pauli sentence.

    Returns:
    --------
    float or complex
        The trace of the Pauli sentence divided by the length of a Pauli string.
    """
    identity = (0,) * len(next(iter(sentence)))
    return sentence.get(identity, 0)

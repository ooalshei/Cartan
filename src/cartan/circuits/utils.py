# Python version 3.11.5
# Created on December 26, 2023
"""
utils
-----
This module contains useful utility functions commonly used when constructing quantum circuits via qiskit.
"""

###### Imports ######
from qiskit import *

#####################


def gate(circ: QuantumCircuit,
         pauli: str | int,
         index: int) -> None:
    """
    Rotates a state that is in the computational basis to either the X or Y basis.

    Parameters:
    -----------
    circ : QuantumCircuit
        The quantum circuit.
    pauli : str | int
        The desired basis.
    index: int
        The qubit to be rotated.
    """
    if pauli == "X" or pauli == 1:
        circ.h(index)
    elif pauli == "Y" or pauli == 2:
        circ.sdg(index)
        circ.h(index)
    elif pauli == "Z" or pauli == 3:
        pass


def inverse_gate(circ: QuantumCircuit,
                 pauli: str | int,
                 index: int) -> None:
    """
    Rotates a state that is in the X or Y basis to the computational basis.

    Parameters:
    -----------
    circ : QuantumCircuit
        The quantum circuit.
    pauli : str | int
        The original basis.
    index : int
        The qubit to be rotated.
    """
    if pauli == "X" or pauli == 1:
        circ.h(index)
    elif pauli == "Y" or pauli == 2:
        circ.h(index)
        circ.s(index)
    elif pauli == "Z" or pauli == 3:
        pass


def measure(circ: QuantumCircuit, string: str | tuple[int, ...]) -> None:
    """
    Measures a Pauli string.

    Parameters:
    -----------
    circ : QuantumCircuit
        The quantum circuit.
    string : str | tuple[int, ...]
        The Pauli string to be measured.
    """
    c = ClassicalRegister(len(string), name='c0')
    circ.add_register(c)
    for i in range(len(string) - 1, -1, -1):
        if string[i] != "-" and string[i] != 0:
            inverse_gate(circ, string[i], i)
            circ.measure(i, i)


def string_exp_product(generators: list[str | tuple[int, ...]],
                       angles: list[float],
                       barrier: bool = False) -> QuantumCircuit:
    r"""
    Generates a circuit for the unitary :math:`\mathrm{e}^{\mathrm{i} x_{1} P_1} ...
    \mathrm{e}^{\mathrm{i} x_n P_n}`.

    Parameters:
    -----------
    generators : list[str | tuple[int, ...]]
        The Pauli strings appearing in the exponents.
    angles : list[float]
        The rotation angles.
    barrier : bool, default=False
        Adds a barrier between the circuits generated by the individual exponentials if True.

    Returns:
    --------
    circ : QuantumCircuit
        The quantum circuit.

    """
    # Consistency check
    if len(angles) != len(generators):
        raise Exception(f"Length mismatch - generators: {len(generators)}, angles: {len(angles)}")

    reversed_angles = angles[::-1]
    reversed_generators = generators[::-1]
    L = len(reversed_generators[0])
    circ = QuantumCircuit(L)

    for i in range(len(reversed_generators)):
        string = reversed_generators[i][::-1]
        unit_circ = QuantumCircuit(L)
        # Rotate the qubits to the appropriate basis
        for j in range(L):
            gate(unit_circ, string[j], j)
        # Look for the first non-identity qubit
        k = L - 1
        while string[k] == "-" or string[k] == 0:
            k -= 1
        # Carry the parity of all the non-identity qubits to the first non-identity one
        for j in range(k):
            if string[j] != "-" and string[j] != 0:
                unit_circ.cx(j, k)

        inverse_circuit = unit_circ.inverse()
        # Apply the rotation
        unit_circ.rz(-2 * reversed_angles[i], k)
        # Disentangle back the qubits
        unit_circ = unit_circ.compose(inverse_circuit)
        if barrier:
            unit_circ.barrier()
        circ = circ.compose(unit_circ)

    return circ


def string_density_matrix(string: str | tuple[int], resets: bool = True) -> QuantumCircuit:
    """
    Prepares the state corresponding to a density matrix that is a single Pauli string.

    Parameters:
    -----------
    string : str | tuple[int, ...]
        The density matrix.
    resets : bool, default=True
        If False, does not use reset gates at the expense of adding more ancilla qubits.

    Returns:
    --------
    circ : QuantumCircuit
        The quantum circuit representing the density matrix.
    """
    L = len(string)
    if resets:
        circ = QuantumCircuit(L + 1)
    elif string == "-" * L or string == (0,) * L:
        circ = QuantumCircuit(2 * L)
    else:
        circ = QuantumCircuit(2 * L - 1)
    reverse_string = string[::-1]

    # Entangle the physical qubits with ancillas until a non-identity element is hit
    for i in range(L):
        if reverse_string[i] == "-" or reverse_string[i] == 0:
            circ.h(i)
            if resets:
                circ.cx(i, L)
                # A reset won't be needed if the string is of the form P----... where P is any non-identity Pauli
                if i < L - 2:
                    circ.reset(L)
                # If the string is identity our job is done
                elif reverse_string[-1] == "-" or reverse_string[-1] == 0:
                    circ.reset(L)
                    circ.h(i + 1)
                    circ.cx(i + 1, L)
                    return circ
            else:
                circ.cx(i, L + i)
                # If the string is identity our job is done
                if i == L - 1:
                    return circ
        # Record the position of the first (from the right) non-identity occurrence
        else:
            j = i
            break

    # If the first non-identity occurrence is the first element in the string our job is done
    if j == L - 1:
        inverse_gate(circ, reverse_string[j], j)
    else:
        for i in range(j + 1, L):
            # Entangle the rest of the qubits
            circ.h(i)
            if resets:
                circ.cx(i, L)
                # A reset won't be needed at the last index
                if i < L - 1:
                    circ.reset(L)
            else:
                circ.cx(i, L + i - 1)
            # Entangle the first non-identity occurrence with the non-identity qubit q_i
            if reverse_string[i] != "-" and reverse_string[i] != 0:
                circ.cx(i, j)
                inverse_gate(circ, reverse_string[i], i)
        inverse_gate(circ, reverse_string[j], j)

    return circ
"""
tfim_circuits
-------------
For Hamiltonians that generate a dynamical Lie algebra like that of the transverse field Ising model, there exists some
identities that drastically reduce the number of CNOT gates required to build a circuit for the time evolution unitary
based on the Cartan decomposition procedure. This module provides circuits based on these simplifications.
"""

from qiskit import *
from .utils import gate, inverse_gate


def xy_gate(circ: QuantumCircuit,
            number_of_sites: int,
            angle1: float,
            angle2: float,
            index1: int,
            index2: int) -> None:
    r"""
    Generates a gate for the unitary :math:`\mathrm{e}^{\mathrm{i}\theta_{1}XY} \mathrm{e}^{\mathrm{i}\theta_{2}YX}`.

    Parameters:
    -----------
    circ : QuantumCircuit
        The quantum circuit.
    number_of_sites : int
        The number of sites of the spin model.
    angle1, angle2 : float
        The angles of the XY and YX rotations, respectively.
    index1, index2 : int
        The two qubits on which the gate is applied. Indices i and j would be :math:`X_i Y_j, Y_i X_j`.
    """
    # Number of sites is important for indexing to avoid any complications that might arise in the presence of ancillas
    # if we immediately reverse the indices.
    i = number_of_sites - 1 - index2
    j = number_of_sites - 1 - index1

    inverse_gate(circ, "Y", i)
    inverse_gate(circ, "Y", j)
    inverse_gate(circ, "X", j)
    circ.cx(i, j)
    circ.rx(-2 * angle1, i)
    circ.rz(-2 * angle2, j)
    circ.cx(i, j)
    gate(circ, "X", j)
    gate(circ, "Y", j)
    gate(circ, "Y", i)


def k_unitary(number_of_sites: int,
              angles: list[float]) -> QuantumCircuit:
    r"""
    Generates the compressed circuit for the K unitary of TFIM-like models. This assumes that the k algebra is chosen
    such that elements are ...--XZZ...Y--.... It takes the form (0,1), (1,2), (2,3), (0,1),
    (3,4), (1,2) ... (N-2, N-1), (N-4, N-3), (N-6, N-5) ... (3,4), (1,2), (2,3), (0,1), (1,2), (0,1), where (i,j) is
    understood to be :math:`\mathrm{e}^{\mathrm{i}\theta_{1}X_i Y_j} \mathrm{e}^{\mathrm{i}\theta_{2}Y_i X_j}`. Using
    the arrow representation this should look like a cascade where each pair of arrows representing (i,j) should have
    two pairs beneath it and only one pair on top of the circuit (i.e., the last two spins).

    Parameters:
    -----------
    circ : QuantumCircuit
        The quantum circuit.
    number_of_sites : int
        The number of sites of the spin model.
    angles : list[float]
        The angles of rotation. The number of angles should be equal to twice the sum over integers from 1 to the
        (number of sites - 1).

    Returns:
    --------
    circ : QuantumCircuit
        The quantum circuit for the unitary.
    """

    # Consistency check
    if len(angles) != 2 * sum(i for i in range(number_of_sites)):
        raise Exception(f"Length mismatch. Number of angles should be {2 * sum(i for i in range(number_of_sites))}.")

    circ = QuantumCircuit(number_of_sites)
    angle_index = 0
    for i in range(number_of_sites - 1):
        for j in range(i, -1, -1):
            xy_gate(circ, number_of_sites, angles[angle_index], angles[angle_index + 1], j, j + 1)
            angle_index += 2
    return circ

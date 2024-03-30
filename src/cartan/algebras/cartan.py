"""
cartan
------
This module contains methods that build Pauli strings given a Hamiltonian, generate the dynamical Lie algebra, and find
Cartan decompositions. The identity along with the three Pauli matrices are referred to as (0, 1, 2, 3).
"""
from typing import Literal
import numpy as np
from .. import pauli_operations


class Hamiltonian:
    """
    This class builds Pauli strings of the Hamiltonians of some known models and constructs the dynamical Lie algebra,
    which is the set of nested commutators of terms in the Hamiltonian. Models currently included are:
        - Transverse field Ising model: XX + Z
        - Transverse field XY model: XX + YY + Z
        - Heisenberg model: XX + YY + ZZ
        - Cross field XY model: XX + YY + Z + Y
        - 6-site UCC model
        - Schwinger model: XX + YY + ZZ + Z
        - Fermions on a ring in a magnetic field: XX + YY + XZ...ZX + YZ...ZY + XZ...ZY - YZ...ZX
        - Creutz model

    Attributes:
    -----------
    number_of_sites : int
        Number of sites.
    model : {"TFIM", "TFXY", "Heisenberg", "CFXY", "UCC", "Schwinger", "fermion_ring", "Creutz"}, optional
        Model name.
    pbc : bool, default=False
        Periodic boundary conditions.
    """

    def __init__(self, number_of_sites: int,
                 model: Literal[
                     "TFIM", "TFXY", "Heisenberg", "CFXY", "UCC", "Schwinger", "fermion_ring", "Creutz", None] = None,
                 pbc: bool = False) -> None:
        self.N = number_of_sites
        self.model = model
        self.pbc = pbc

    def builder(self, parameters: list[float | int, ...] | None = None) -> tuple[list[tuple[int, ...]], list[float]]:
        """
        Builds Pauli strings for the specified model. Returns an empty list if no model is specified.

        Parameters:
        -----------
        parameters: any, optional
            The model parameters. If not specified, ferromagnetism is assumed with J = 1.

        Returns:
        --------
        string_list: list[tuple[int, ...], ...] | list[]
            List of Hamiltonian Pauli strings.
        coefficient_list: list[float, ...]
            List of coefficients. Both lists should be of tha same length. Each term corresponds to the respective term
            in string_list.
        """
        if self.model == 'TFIM':
            string_list = []
            coefficient_list = []
            coefficients = [1, 1] if parameters is None else parameters
            coefficient_xx = -coefficients[0]
            coefficient_z = -coefficients[0] * coefficients[1]

            if coefficient_xx != 0:
                for i in range(self.N - 1):
                    string = [0] * self.N
                    string[i] = 1
                    string[i + 1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_xx)
                if self.pbc:
                    string = [0] * self.N
                    string[0] = 1
                    string[-1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_xx)

            if coefficient_z != 0:
                for i in range(self.N):
                    string = [0] * self.N
                    string[i] = 3
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_z)

            return string_list, coefficient_list

        elif self.model == 'TFXY':
            string_list = []
            coefficient_list = []
            coefficients = [1, 1] if parameters is None else parameters
            coefficient_xy = -coefficients[0]
            coefficient_z = -coefficients[0] * coefficients[1]

            if coefficient_xy != 0:
                for i in range(self.N - 1):
                    string = [0] * self.N
                    string[i] = 1
                    string[i + 1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_xy)

                    string = [0] * self.N
                    string[i] = 2
                    string[i + 1] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_xy)

                    if self.pbc:
                        string = [0] * self.N
                        string[0] = 1
                        string[-1] = 1
                        string_list.append(tuple(string))
                        coefficient_list.append(coefficient_xy)

                        string = [0] * self.N
                        string[0] = 2
                        string[-1] = 2
                        string_list.append(tuple(string))
                        coefficient_list.append(coefficient_xy)

            if coefficient_z != 0:
                for i in range(self.N):
                    string = [0] * self.N
                    string[i] = 3
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_z)

            return string_list, coefficient_list

        elif self.model == 'Heisenberg':
            string_list = []
            coefficients = -1 if parameters is None else -parameters[0]
            coefficient_list = [coefficients] * 3 * (self.N - 1)

            for i in range(self.N - 1):
                string = [0] * self.N
                string[i] = 1
                string[i + 1] = 1
                string_list.append(tuple(string))

                string = [0] * self.N
                string[i] = 2
                string[i + 1] = 2
                string_list.append(tuple(string))

                string = [0] * self.N
                string[i] = 3
                string[i + 1] = 3
                string_list.append(tuple(string))

            if self.pbc:
                string = [0] * self.N
                string[0] = 1
                string[-1] = 1
                string_list.append(tuple(string))
                coefficient_list.append(coefficients)

                string = [0] * self.N
                string[0] = 2
                string[-1] = 2
                string_list.append(tuple(string))
                coefficient_list.append(coefficients)

                string = [0] * self.N
                string[0] = 3
                string[-1] = 3
                string_list.append(tuple(string))
                coefficient_list.append(coefficients)

            return string_list, coefficient_list

        elif self.model == 'CFXY':
            string_list = []
            coefficient_list = []
            coefficients = [1, 1, 1] if parameters is None else parameters
            coefficient_xy = -coefficients[0]
            coefficient_z = -coefficients[0] * coefficients[1]
            coefficient_y = -coefficients[0] * coefficients[2]

            if coefficient_xy != 0:
                for i in range(self.N - 1):
                    string = [0] * self.N
                    string[i] = 1
                    string[i + 1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_xy)

                    string = [0] * self.N
                    string[i] = 2
                    string[i + 1] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_xy)

                    if self.pbc:
                        string = [0] * self.N
                        string[0] = 1
                        string[-1] = 1
                        string_list.append(tuple(string))
                        coefficient_list.append(coefficient_xy)

                        string = [0] * self.N
                        string[0] = 2
                        string[-1] = 2
                        string_list.append(tuple(string))
                        coefficient_list.append(coefficient_xy)

            if coefficient_z != 0:
                for i in range(self.N):
                    string = [0] * self.N
                    string[i] = 3
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_z)

            if coefficient_y != 0:
                for i in range(self.N):
                    string = [0] * self.N
                    string[i] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(coefficient_y)

            return string_list, coefficient_list

        elif self.model == 'UCC':

            string_list = [(1, 1, 1, 3, 3, 2), (2, 2, 1, 3, 3, 2), (1, 2, 2, 3, 3, 2), (2, 1, 2, 3, 3, 2),
                           (1, 2, 1, 3, 3, 1),
                           (2, 1, 1, 3, 3, 1), (1, 1, 2, 3, 3, 1), (2, 2, 2, 3, 3, 1),
                           (1, 1, 0, 1, 2, 0), (2, 2, 0, 1, 2, 0), (1, 2, 0, 1, 1, 0), (2, 1, 0, 1, 1, 0),
                           (1, 2, 0, 2, 2, 0),
                           (2, 1, 0, 2, 2, 0), (1, 1, 0, 2, 1, 0), (2, 2, 0, 2, 1, 0)]

            coefficient_list = [1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1]

            return string_list, coefficient_list

        elif self.model == 'Schwinger':

            string_list = []
            coefficient_list = []
            coefficients = [1, 1] if parameters is None else parameters

            def double(n):
                return (self.N - n) * (self.N - n + 1)

            def single(n):
                return -2 * (np.floor((self.N - 1) ** 2 / 4) - np.floor((n - 1) / 2) * (self.N - 1) + (
                        n - 1) ** 2 / 4 + ((n - 1) % 2) * (1 - 2 * (n - 1)) / 4)

            for i in range(self.N - 1):

                string = [0] * self.N
                string[i] = 1
                string[i + 1] = 1
                string_list.append(tuple(string))
                coefficient_list.append(coefficients[0])

                string = [0] * self.N
                string[i] = 2
                string[i + 1] = 2
                string_list.append(tuple(string))
                coefficient_list.append(coefficients[0])

                c = 0.25 * double(i)
                for j in range(i):
                    string = [0] * self.N
                    string[i] = 3
                    string[j] = 3
                    string_list.append(tuple(string))
                    coefficient_list.append(c)

                c = 0.25 * single(i) + (-1) ** (i - 1) * coefficients[1] / 2
                string = [0] * self.N
                string[i] = 3
                string_list.append(tuple(string))
                coefficient_list.append(c)

            string = [0] * self.N
            string[self.N - 1] = 3
            c = (-1) ** (self.N - 1) * coefficients[1] / 2
            string_list.append(tuple(string))
            coefficient_list.append(c)

            return string_list, coefficient_list

        elif self.model == "fermion_ring":

            string_list = []
            coefficient_list = []
            coefficients = [1, np.pi / 4] if parameters is None else parameters

            for i in range(self.N - 1):
                string = [0] * self.N
                string[i] = 1
                string[i + 1] = 1
                string_list.append(tuple(string))
                coefficient_list.append(-coefficients[0] / 2)

                string = [0] * self.N
                string[i] = 2
                string[i + 1] = 2
                string_list.append(tuple(string))
                coefficient_list.append(-coefficients[0] / 2)

            string = [3] * self.N
            string[0] = 1
            string[-1] = 1
            string_list.append(tuple(string))
            coefficient_list.append(-coefficients[0] * np.cos(coefficients[1]) / 2)

            string = [3] * self.N
            string[0] = 2
            string[-1] = 2
            string_list.append(tuple(string))
            coefficient_list.append(-coefficients[0] * np.cos(coefficients[1]) / 2)

            string = [3] * self.N
            string[0] = 1
            string[-1] = 2
            string_list.append(tuple(string))
            coefficient_list.append(-coefficients[0] * np.sin(coefficients[1]) / 2)

            string = [3] * self.N
            string[0] = 2
            string[-1] = 1
            string_list.append(tuple(string))
            coefficient_list.append(coefficients[0] * np.sin(coefficients[1]) / 2)

            return string_list, coefficient_list

        elif self.model == "Creutz":
            string_list = []
            coefficient_list = []
            coefficients = [1, 1, 1, np.pi / 4] if parameters is None else parameters
            t_cos = -coefficients[0] / 2 * np.cos(coefficients[3])
            t_sin = -coefficients[0] / 2 * np.sin(coefficients[3])
            t_d = -coefficients[1] / 2
            t_v = -coefficients[2] / 2

            if np.abs(t_cos) >= 1e-10:
                for i in range(2 * self.N - 2):
                    string = [0] * 2 * self.N
                    string[i] = 1
                    string[i + 1] = 3
                    string[i + 2] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(t_cos)

                    string = [0] * 2 * self.N
                    string[i] = 2
                    string[i + 1] = 3
                    string[i + 2] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(t_cos)

                if self.pbc:
                    string = [3] * 2 * self.N
                    string[0] = 1
                    string[-2] = 1
                    string[-1] = 0
                    string_list.append(tuple(string))
                    coefficient_list.append(t_cos)

                    string = [3] * 2 * self.N
                    string[0] = 2
                    string[-2] = 2
                    string[-1] = 0
                    string_list.append(tuple(string))
                    coefficient_list.append(t_cos)

                    string = [3] * 2 * self.N
                    string[0] = 0
                    string[1] = 1
                    string[-1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(t_cos)

                    string = [3] * 2 * self.N
                    string[0] = 0
                    string[1] = 2
                    string[-1] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(t_cos)

            if np.abs(t_sin) >= 1e-10:
                for i in range(2 * self.N - 2):
                    string = [0] * 2 * self.N
                    string[i] = 1
                    string[i + 1] = 3
                    string[i + 2] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append((-1) ** i * t_sin)

                    string = [0] * 2 * self.N
                    string[i] = 2
                    string[i + 1] = 3
                    string[i + 2] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(-(-1) ** i * t_sin)

                if self.pbc:
                    string = [3] * 2 * self.N
                    string[0] = 1
                    string[-2] = 2
                    string[-1] = 0
                    string_list.append(tuple(string))
                    coefficient_list.append(-t_sin)

                    string = [3] * 2 * self.N
                    string[0] = 2
                    string[-2] = 1
                    string[-1] = 0
                    string_list.append(tuple(string))
                    coefficient_list.append(t_sin)

                    string = [3] * 2 * self.N
                    string[0] = 0
                    string[1] = 1
                    string[-1] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(t_sin)

                    string = [3] * 2 * self.N
                    string[0] = 0
                    string[1] = 2
                    string[-1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(-t_sin)

            if t_d != 0:
                for i in range(2 * self.N - 2):
                    if i % 2 == 0:
                        string = [0] * 2 * self.N
                        string[i] = 1
                        string[i + 1] = 3
                        string[i + 2] = 3
                        string[i + 3] = 1
                        string_list.append(tuple(string))
                        coefficient_list.append(t_d)

                        string = [0] * 2 * self.N
                        string[i] = 2
                        string[i + 1] = 3
                        string[i + 2] = 3
                        string[i + 3] = 2
                        string_list.append(tuple(string))
                        coefficient_list.append(t_d)

                    else:
                        string = [0] * 2 * self.N
                        string[i] = 1
                        string[i + 1] = 1
                        string_list.append(tuple(string))
                        coefficient_list.append(t_d)

                        string = [0] * 2 * self.N
                        string[i] = 2
                        string[i + 1] = 2
                        string_list.append(tuple(string))
                        coefficient_list.append(t_d)

                if self.pbc:
                    string = [3] * 2 * self.N
                    string[0] = 1
                    string[-1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(t_d)

                    string = [3] * 2 * self.N
                    string[0] = 2
                    string[-1] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(t_d)

                    string = [3] * 2 * self.N
                    string[0] = 0
                    string[1] = 1
                    string[-2] = 1
                    string[-1] = 0
                    string_list.append(tuple(string))
                    coefficient_list.append(t_d)

                    string = [3] * 2 * self.N
                    string[0] = 0
                    string[1] = 2
                    string[-2] = 2
                    string[-1] = 0
                    string_list.append(tuple(string))
                    coefficient_list.append(t_d)

            if t_v != 0:
                for i in range(0, 2 * self.N - 1, 2):
                    string = [0] * 2 * self.N
                    string[i] = 1
                    string[i + 1] = 1
                    string_list.append(tuple(string))
                    coefficient_list.append(t_v)

                    string = [0] * 2 * self.N
                    string[i] = 2
                    string[i + 1] = 2
                    string_list.append(tuple(string))
                    coefficient_list.append(t_v)

            return string_list, coefficient_list

        else:
            return []  # type: ignore

    def algebra(self, hamiltonian_list: list[tuple[int, ...]] | None = None) -> list[tuple[int, ...]]:
        """
        Constructs the dynamical Lie algebra. This is the nested commutators of the individual Hamiltonian terms.

        Parameters:
        -----------
        hamiltonian_list : list[tuple[int, ...]] | None, optional
            Strings representing the Hamiltonian. If not provided, the object model will be assumed.

        Returns:
        --------
        algebra_list : list[tuple[int, ...]]
            Strings representing the dynamical Lie algebra.
        """

        algebra_list = self.builder()[0].copy() if hamiltonian_list is None else hamiltonian_list.copy()
        final_index = len(algebra_list) - 1
        initial_index = -1
        t = True

        while t:
            t = False
            for i in range(final_index, initial_index, -1):
                for j in range(i - 1, -1, -1):
                    string, s, c = pauli_operations.string_product(algebra_list[i], algebra_list[j])

                    if not c:
                        if string not in algebra_list:
                            t = True
                            algebra_list.append(string)

            initial_index = final_index
            final_index = len(algebra_list) - 1

        return algebra_list


class CartanDecomposition(Hamiltonian):
    r"""
    This class finds a Cartan decomposition of a provided algebra. A Cartan decomposition of a semisimple Lie algebra is
    an orthogonal split :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` such that
    .. math::
        [\mathfrak{k}, \mathfrak{k}]\subseteq \mathfrak{k}, \qquad [\mathfrak{m}, \mathfrak{m}]\subseteq \mathfrak{k},
        \qquad [\mathfrak{k}, \mathfrak{m}]\subseteq \mathfrak{m}.
    A Cartan subalgebra :math:`\mathfrak{h}` is a maximal Abelian subalgebra in :math:`\mathfrak{m}`.
    To find a Cartan decomposition, one can use an involution which would immediately yield an orthogonal split:
    :math:`\Theta (\mathfrak{k}) = \mathfrak{k}` and :math:`\Theta (\mathfrak{m}) = -\mathfrak{m}`. Currently only one
    involution is included, which counts the number of Y in a Pauli string.

    Attributes:
    -----------
    number_of_sites : int
        Number of sites.
    model : {"TFIM", "TFXY", "Heisenberg", "CFXY", "UCC", "Schwinger", "fermion_ring", "Creutz"}, optional
        Model name.
    pbc : bool, default=False
    """

    def __init__(self, number_of_sites: int,
                 model: Literal[
                     "TFIM", "TFXY", "Heisenberg", "CFXY", "UCC", "Schwinger", "fermion_ring", "Creutz", None] = None,
                 pbc: bool = False) -> None:
        super().__init__(number_of_sites, model, pbc)

    def decomposition(self, algebra_list: list[tuple[int, ...]] | None = None, involution: str = "even_odd") -> tuple[
        list[tuple[int, ...]], list[tuple[int, ...]], list[tuple[int, ...]]]:
        r"""
        Finds a Cartan decomposition. This assumes that we can find a decomposition such that
        :math:`\mathcal{H} \subset \mathfrak{m}`.

        Parameters:
        -----------
        algebra_list : list[tuple[int, ...]], optional
            List of Pauli strings to decompose. If not provided, the object model will be assumed.
        involution : str, default="even_odd"
            The involution to be used.

        Returns:
        --------
        k_strings : list[tuple[int, ...]]
            List of strings in :math:`\mathfrak{k}`.
        m_strings : list[tuple[int, ...]]
            List of strings in :math:`\mathfrak{m}`.
        subalgebra_strings : list[tuple[int, ...]]
            List of strings in :math:`\mathfrak{h}`.
        """

        if algebra_list is None:
            algebra_list = self.algebra()
        sorting_list = [self.N - string.count(0) for string in algebra_list]
        sorted_algebra_list = [string for _, string in sorted(zip(sorting_list, algebra_list))]
        k_strings = []
        m_strings = []
        subalgebra_strings = [(0,) * self.N]

        if involution == "even_odd":
            for string in sorted_algebra_list:
                if string.count(2) % 2 == 0:
                    k_strings.append(string)
                else:
                    m_strings.append(string)
                    for h in subalgebra_strings:
                        c = pauli_operations.string_product(string, h)[2]
                        if c:
                            subalgebra_strings.append(string)
        subalgebra_strings.pop(0)

        return k_strings, m_strings, subalgebra_strings

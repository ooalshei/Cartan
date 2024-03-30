r"""
involutionless_cartan
------
This module finds a Cartan decomposition such that :math:`\mathcal{H} \subset \mathfrak{m}` without the need for an
involution. This decomposition is unique, and if no such decomposition is found, the Hamiltonian is said to be
non-Cartan. The identity along with the three Pauli matrices are referred to as (0, 1, 2, 3).
"""
from typing import Literal
from .cartan import CartanDecomposition
from .. import pauli_operations


class InvolutionlessCartan(CartanDecomposition):
    r"""
    This class finds a Cartan decomposition of a provided algebra. A Cartan decomposition of a semisimple Lie algebra is
    an orthogonal split :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` such that
    .. math::
        [\mathfrak{k}, \mathfrak{k}]\subseteq \mathfrak{k}, \qquad [\mathfrak{m}, \mathfrak{m}]\subseteq \mathfrak{k},
        \qquad [\mathfrak{k}, \mathfrak{m}]\subseteq \mathfrak{m}.
    A Cartan subalgebra :math:`\mathfrak{h}` is a maximal Abelian subalgebra in :math:`\mathfrak{m}`. If we choose
    :math:`\mathcal{H} \subset \mathfrak{m}`, the decomposition becomes unique and we can easily find it, if it exists.
    If the Hamiltonian is non-Cartan, an Abelian subalgebra in :math:`\mathfrak{g}` is chosen instead to be our Cartan
    subalgebra.

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

    def decomposition(self, hamiltonian_list: list[tuple[int, ...]] | None = None, involution=None) -> dict[
        str, list[tuple[int, ...]] | bool]:
        r"""
        Finds an involutionless Cartan decomposition keeping :math:`\mathfrak{m}`. Associating +1 (-1) to an element in
        :math:`\mathfrak{k}` (:math:`\mathfrak{m}`), we can easily decompose our generated dynamical Lie algebra by
        keeping track of this sign. When two elements are commuted, their signs are multiplied.

        Parameters:
        -----------
        hamiltonian_list : list[tuple[int, ...]], optional
            List of Pauli strings in the Hamiltonian. If not provided, the object model will be assumed.
        involution : None

        Returns:
        --------
        algebra_list : list[tuple[int]]
            Dynamical Lie algebra.
        contradicton : bool
            True if Hamiltonian is non-Cartan. False otherwise.
        generator_strings : list[tuple[int]]
            If Hamiltonian is non-Cartan, this list will contain the DLA - Cartan subalgebra (to be used later when
            finding the optimal angles).
        k_strings : list[tuple[int]]
            If Hamiltonian is Cartan, this is the list of strings in :math:`\mathfrak{k}`.
        m_strings : list[tuple[int]]
            If Hamiltonian is Cartan, this is the list of strings in :math:`\mathfrak{m}`.
        subalgebra_strings : list[tuple[int]]
            List of strings in :math:`\mathfrak{h}`.
        """

        algebra_list = self.builder()[0].copy() if hamiltonian_list is None else hamiltonian_list.copy()
        sign_list = [-1] * len(algebra_list)
        k_strings = []
        m_strings = algebra_list.copy()
        final_index = len(algebra_list) - 1
        initial_index = -1
        t = True
        contradiction = False

        while t:
            t = False
            for i in range(final_index, initial_index, -1):
                for j in range(i - 1, -1, -1):
                    string, _, c = pauli_operations.string_product(algebra_list[i], algebra_list[j])
                    sign = sign_list[i] * sign_list[j]

                    if not c:
                        if string not in algebra_list:
                            t = True
                            algebra_list.append(string)
                            sign_list.append(sign)
                            k_strings.append(string) if sign == 1 else m_strings.append(string)

                        elif sign != sign_list[algebra_list.index(string)]:
                            contradiction = True

            initial_index = final_index
            final_index = len(algebra_list) - 1

        if contradiction:
            sorting_list = [self.N - string.count(0) for string in algebra_list]
            sorted_algebra_list = [string for _, string in sorted(zip(sorting_list, algebra_list))]
            subalgebra_strings = [sorted_algebra_list[0]]
            generator_strings = algebra_list.copy()
            generator_strings.remove(subalgebra_strings[0])

            for g in sorted_algebra_list[1:]:
                for h in subalgebra_strings:
                    c = pauli_operations.string_product(g, h)[2]
                    if not c:
                        break
                    elif h == subalgebra_strings[-1]:
                        subalgebra_strings.append(g)
                        generator_strings.remove(g)
                        break

            return {"DLA": algebra_list, "contradiction": contradiction, "g": generator_strings,
                    "h": subalgebra_strings}

        else:
            sorting_list = [self.N - string.count(0) for string in m_strings]
            sorted_m_strings = [string for _, string in sorted(zip(sorting_list, m_strings))]
            subalgebra_strings = [sorted_m_strings[0]]

            for m in sorted_m_strings[1:]:
                for h in subalgebra_strings:
                    c = pauli_operations.string_product(m, h)[2]
                    if not c:
                        break
                    elif h == subalgebra_strings[-1]:
                        subalgebra_strings.append(m)
                        break

            return {"DLA": algebra_list, "contradiction": contradiction, "k": k_strings, "m": m_strings,
                    "h": subalgebra_strings}

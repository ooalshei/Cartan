from .. import pauli_operations
from .cartan import CartanDecomposition


class InvolutionlessCartan(CartanDecomposition):

    def __init__(self, number_of_sites, model):
        super().__init__(number_of_sites, model)

    def decomposition(self, hamiltonian_list=None, involution=None):

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

        sorting_list = [self.N - string.count(0) for string in algebra_list]
        if contradiction:
            sorted_algebra_list = [string for _, string in sorted(zip(sorting_list, algebra_list))]
            subalgebra_strings = [sorted_algebra_list[0]]

            for g in sorted_algebra_list[1:]:
                for h in subalgebra_strings:
                    c = pauli_operations.string_product(g, h)[2]
                    if not c:
                        break
                    elif h == subalgebra_strings[-1]:
                        subalgebra_strings.append(g)
                        break

            return {"DLA": algebra_list, "h": subalgebra_strings, "contradiction": contradiction}

        else:
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

            return {"DLA": algebra_list, "k": k_strings, "m": m_strings, "h": subalgebra_strings,
                    "contradiction": contradiction}

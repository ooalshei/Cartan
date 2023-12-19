# Python version 3.11.5
# Created on December 18, 2023


################## Imports ##################
import numpy as np

from src.cartan.pauli_helpers import pauli_operations


#############################################

class Hamiltonian:

    def __init__(self, number_of_sites, model=None):
        self.N = number_of_sites
        self.model = model

    def builder(self, coefficients=None):

        if self.model == 'TFIM':
            string_list = []
            coefficient_list = []
            x = [1] * self.N if coefficients is None else coefficients

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))
                coefficient_list.append(-1)

                l = [0] * self.N
                l[i] = 3
                string_list.append(tuple(l))
                coefficient_list.append(x[i])

            l = [0] * self.N
            l[self.N - 1] = 3
            string_list.append(tuple(l))
            coefficient_list.append(x[-1])

            return string_list, coefficient_list

        elif self.model == 'XY':
            string_list = []
            coefficient_list = [1] * 2 * (self.N - 1)

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))

            return string_list, coefficient_list

        elif self.model == 'TFXY':
            string_list = []
            coefficient_list = []
            x = [1] * self.N if coefficients is None else coefficients

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))
                coefficient_list.append(1)

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))
                coefficient_list.append(1)

                l = [0] * self.N
                l[i] = 3
                string_list.append(tuple(l))
                coefficient_list.append(x[i])

            l = [0] * self.N
            l[self.N - 1] = 3
            string_list.append(tuple(l))
            coefficient_list.append(x[-1])

            return string_list, coefficient_list

        elif self.model == 'TFXYY':
            string_list = []

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))

                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 2
                string_list.append(tuple(l))

                l = [0] * self.N
                l[i] = 3
                string_list.append(tuple(l))

            l = [0] * self.N
            l[self.N - 1] = 3
            string_list.append(tuple(l))

            return string_list

        elif self.model == 'Heisenberg':
            string_list = []
            coefficient_list = [1] * 3 * (self.N - 1)

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))

                l = [0] * self.N
                l[i] = 3
                l[i + 1] = 3
                string_list.append(tuple(l))

            return string_list, coefficient_list

        elif self.model == 'CFXY':
            string_list = []
            coefficient_list = []
            x = [(1, 1)] * self.N if coefficients is None else coefficients

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))
                coefficient_list.append(1)

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))
                coefficient_list.append(1)

                l = [0] * self.N
                l[i] = 3
                string_list.append(tuple(l))
                coefficient_list.append(x[i][0])

                l = [0] * self.N
                l[i] = 2
                string_list.append(tuple(l))
                coefficient_list.append(x[i][1])

            l = [0] * self.N
            l[self.N - 1] = 3
            string_list.append(tuple(l))
            coefficient_list.append(x[-1][0])

            l = [0] * self.N
            l[self.N - 1] = 2
            string_list.append(tuple(l))
            coefficient_list.append(x[-1][1])

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
            x = [1, 1] if coefficients is None else coefficients

            def double(n):
                return (self.N - n) * (self.N - n + 1)

            def single(n):
                return -2 * (np.floor((self.N - 1) ** 2 / 4) - np.floor((n - 1) / 2) * (self.N - 1) + (
                        n - 1) ** 2 / 4 + ((n - 1) % 2) * (1 - 2 * (n - 1)) / 4)

            for i in range(self.N - 1):

                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))
                coefficient_list.append(x[0])

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))
                coefficient_list.append(x[0])

                c = 0.25 * double(i)
                for j in range(i):
                    l = [0] * self.N
                    l[i] = 3
                    l[j] = 3
                    string_list.append(tuple(l))
                    coefficient_list.append(c)

                c = 0.25 * single(i) + (-1) ** (i - 1) * x[1] / 2
                l = [0] * self.N
                l[i] = 3
                string_list.append(tuple(l))
                coefficient_list.append(c)

            l = [0] * self.N
            l[self.N - 1] = 3
            c = (-1) ** (self.N - 1) * x[1] / 2
            string_list.append(tuple(l))
            coefficient_list.append(c)

            return string_list, coefficient_list

        elif self.model == "fermion_ring":

            string_list = []
            coefficient_list = []
            parameters = [1, np.pi / 4] if coefficients is None else coefficients.copy()

            for i in range(self.N - 1):
                l = [0] * self.N
                l[i] = 1
                l[i + 1] = 1
                string_list.append(tuple(l))
                coefficient_list.append(-parameters[0] / 2)

                l = [0] * self.N
                l[i] = 2
                l[i + 1] = 2
                string_list.append(tuple(l))
                coefficient_list.append(-parameters[0] / 2)

            l = [3] * self.N
            l[0] = 1
            l[-1] = 1
            string_list.append(l)
            coefficient_list.append(-parameters[0] * np.cos(parameters[1]) / 2)

            l = [3] * self.N
            l[0] = 2
            l[-1] = 2
            string_list.append(l)
            coefficient_list.append(-parameters[0] * np.cos(parameters[1]) / 2)

            l = [3] * self.N
            l[0] = 1
            l[-1] = 2
            string_list.append(l)
            coefficient_list.append(-parameters[0] * np.sin(parameters[1]) / 2)

            l = [3] * self.N
            l[0] = 2
            l[-1] = 1
            string_list.append(l)
            coefficient_list.append(parameters[0] * np.sin(parameters[1]) / 2)

            return string_list, coefficient_list

        else:
            return []

    def algebra(self, hamiltonian_list=None):

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

    def __init__(self, number_of_sites, model):
        super().__init__(number_of_sites, model)

    def decomposition(self, algebra_list=None, involution="even_odd"):

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

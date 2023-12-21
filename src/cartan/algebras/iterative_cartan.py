# Python version 3.11.5
# Modified on December 20, 2023


############# Imports #############
from .. import pauli_operations
from . import InvolutionlessCartan

###################################


class IterativeCartan(InvolutionlessCartan):

    def __init__(self, number_of_sites, model):
        super().__init__(number_of_sites, model)

    def abelian_subalgebra(self, subalgebra=None):

        subalgebra_list = self.decomposition(subalgebra)["h"] if subalgebra is None else subalgebra.copy()
        abelian_strings = [subalgebra_list[0]]
        multiplication_closure = abelian_strings.copy()

        i = 1
        while not all(string in multiplication_closure for string in subalgebra_list):
            while True:
                subalgebra_string = subalgebra_list[i]
                if subalgebra_string not in multiplication_closure:
                    abelian_strings.append(subalgebra_string)
                    i += 1
                    break
                else:
                    i += 1

            for j in range(len(multiplication_closure)):
                string = pauli_operations.string_product(subalgebra_string, multiplication_closure[j])[0]
                multiplication_closure.append(string)
            multiplication_closure.append(subalgebra_string)

        return abelian_strings

    def symmetric_subspace(self, algebra=None, abelian_subalgebra=None):

        abelian_strings = self.abelian_subalgebra() if abelian_subalgebra is None else abelian_subalgebra.copy()
        if algebra is None:
            decomposition = self.decomposition()
            algebra_strings = decomposition["DLA"] if decomposition["contradiction"] else decomposition["k"]
        else:
            algebra_strings = algebra.copy()

        subspace_list = []
        for string in abelian_strings:
            subspace_strings = []
            i = 0
            while i < len(algebra_strings):
                c = pauli_operations.string_product(string, algebra_strings[i])[2]
                if not c:
                    subspace_strings.append(algebra_strings.pop(i))
                else:
                    i += 1

            subspace_list.append(subspace_strings)
        return subspace_list

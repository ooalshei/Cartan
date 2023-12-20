# Python version 3.11.5
# Created on December 18, 2023


from involutionless_cartan import InvolutionlessCartan
####################### Imports #######################
from src.cartan import pauli_operations


#######################################################


class IterativeCartan(InvolutionlessCartan):

    def __init__(self, number_of_sites, model):
        super().__init__(number_of_sites, model)

    def abelian_subalgebra(self, subalgebra=None):

        subalgebra_list = self.decomposition(subalgebra)["h"] if subalgebra is None else subalgebra.copy()
        abelian_strings = [subalgebra_list[0], subalgebra_list[1]]
        for i in range(len(subalgebra_list)):
            for j in range(i + 1, len(subalgebra_list)):
                string = pauli_operations.string_product(subalgebra_list[i], subalgebra_list[j])[0]
                if string not in abelian_strings:
                    abelian_strings.append(string)

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
                c = pauli_operations.string_product(string, algebra_strings[i])
                if not c:
                    subspace_strings.append(algebra_strings.pop(i))
                else:
                    i += 1

            subspace_list.append(subspace_strings)
        return subspace_list

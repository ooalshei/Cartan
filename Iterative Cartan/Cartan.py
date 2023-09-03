class Pauli:
    
    def __init__(self):
        self.rules = [1,3,1,3]
        self.sign_rules = [[1,1,1,1], 
             [1, 1, 1j, -1j],
             [1, -1j, 1, 1j],
             [1, 1j, -1j, 1]]
        
        
    def Product(self, a, b):
        p = (a + b*self.rules[a])%4
        return p
    
    
    def Commutator(self, A, B):
    
        C = ()

        forwardsign = 1
        backwardsign = 1

        for i in range(len(A)):
            C += (self.Product(A[i],B[i]),)
            forwardsign = forwardsign * self.sign_rules[A[i]][B[i]]
            backwardsign = backwardsign * self.sign_rules[B[i]][A[i]]

        if forwardsign == backwardsign:
            return 0

        else:
            return C
        
        
    def letters(self, A):
        
        label = {0: "I", 1: "X", 2: "Y", 3: "Z"}
        C=[]
        
        for i in range(len(A)):
            B=""
            for j in range(len(A[i])):
                B += label[A[i][j]]
            C.append(B)
            
        return C



class Cartan:
    
    
    def __init__(self, N, model = None):
        self.N = N
        self.model = model
        
        
    def Hamiltonian(self):
    
        if self.model == 'TFIM':
            H = []
            #l = [0]*self.N

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))

            return H
        
           
        elif self.model == 'XY':
            H = []
            #l = [0]*self.N

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))

            return H 
        
        
        elif self.model == 'TFXY':
            H = []
            #l = [0]*self.N

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))

            return H
        
        
        elif self.model == 'TFXYY':
            H = []
            #l = [0]*self.N

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 1
                l[i+1] = 2
                H.append(tuple(l))

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))

            return H
        
        
        elif self.model == 'Heisenberg':
            H = []
            #l = [0]*self.N

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))

                l = [0]*self.N
                l[i] = 3
                l[i+1] = 3
                H.append(tuple(l))
                
            return H
        
        
        elif self.model == 'CFXY':
            H = []
            #l = [0]*self.N

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))
                
                l = [0]*self.N
                l[i] = 2
                H.append(tuple(l))

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))
            
            l = [0]*self.N
            l[self.N-1] = 2
            H.append(tuple(l))

            return H
   

        else:
            
            return []
    
    
    def Algebra(self):
        
        g = self.Hamiltonian()
        s = [-1]*len(g)
        finalindex = len(g) - 1
        initialindex = -1
        P = Pauli()
        t = True
        cont = False
        
        while t == True:
            t = False
            
            for i in range(finalindex, initialindex, -1):
                for j in range(i-1, -1, -1):
                    Com = P.Commutator(g[i],g[j])
                    sign = s[i]*s[j]

                    if Com != 0:
                        
                        if Com not in g:
                            g.append(Com)
                            s.append(sign)
                            t = True
                            
                        elif sign != s[g.index(Com)]:
                            cont = True
                            
                        
                        
            initialindex = finalindex
            finalindex = len(g) - 1
            
        return g, s, cont
        
        
    def decomposition(self):
        
        A = self.Algebra()[0]
        k = []
        m = []
        #c = []
        
        for i in range(len(A)):
            if (A[i].count(2))%2 != 0:
                k.append(A[i])
                
            else:
                m.append(A[i])
                #c.append(A[i].count(0))
                
        return k, m#, np.argmax(c)
    
    
    def subalgebra(self):
        
        m = self.decomposition()[1]
        c = []

        for i in range(len(m)):
            c.append(self.N - m[i].count(0))

        m_sorted = [x for _,x in sorted(zip(c, m))]

            
        h = [m_sorted[0]]
        P = Pauli()
        
        for i in range(1,len(m_sorted)):
            for j in range(len(h)):
                
                if P.Commutator(m_sorted[i],h[j]) != 0:
                    break
                    
                elif j == len(h)-1:
                    h.append(m_sorted[i])
                    
        return h
 
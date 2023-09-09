import numpy as np

class Pauli:
    
    def __init__(self):
        self.rules = [1,3,1,3]
        self.sign_rules = [[1,1,1,1], 
             [1, 1, 1j, -1j],
             [1, -1j, 1, 1j],
             [1, 1j, -1j, 1]]
        
        
    def product(self, a, b):
        p = (a + b*self.rules[a])%4
        return p
    
    
    def commutator(self, A, B):
    
        C = ()

        forwardsign = 1
        backwardsign = 1

        for i in range(len(A)):
            C += (self.product(A[i],B[i]),)
            forwardsign = forwardsign * self.sign_rules[A[i]][B[i]]
            backwardsign = backwardsign * self.sign_rules[B[i]][A[i]]

        if forwardsign == backwardsign:
            return 0

        else:
            return C
        
        
    def letters(self, A):
        
        label = {0: "I", 1: "X", 2: "Y", 3: "Z"}
        C=[]

        if A == []:

            pass
        
        elif type(A[0][1]) != tuple:
        
            for i in range(len(A)):
                B=""
                for j in range(len(A[i])):
                    B += label[A[i][j]]
                C.append(B)

        else:

            D=[]
            a=[]

            for i in range(len(A)):
                B=""
                a.append(A[i][0])
                for j in range(len(A[i][-1])):
                    B += label[A[i][-1][j]]
                D.append(B)
            
            C = list(zip(a,D))
            
        print(C)


class functions(Pauli):

    
    def __init__(self):
        Pauli.__init__(self)


    def mut_irr(self, n, x = np.pi):

        y = x%1
    
        if n > 1:
            return self.mut_irr(n-1, np.pi*y)
    
        else:
            return y


    def prepare(self, a, A):

        if type(A) == tuple:
            
            return [(a, A)]

        else:
            
            X = list(zip(a, A))
                
            return X

    
    def fullprod(self, Xp, Yp):

        prelist = []
        Alist = []
        Z = []
        
        for i in range(len(Xp)):
            for j in range(len(Yp)):

                A = ()
                prefactor = Xp[i][0] * Yp[j][0]
                
                for k in range(len(Xp[i][1])):
                    
                    A += (self.product(Xp[i][1][k], Yp[j][1][k]),)
                    prefactor = prefactor * self.sign_rules[Xp[i][1][k]][Yp[j][1][k]]

                if A not in Alist:
                    Alist.append(A)
                    prelist.append(prefactor)
                    

                else:
                    index = Alist.index(A)
                    prefactor += prelist[index]

                    if abs(prefactor) != 0:
                        prelist[index] = prefactor

                    else:
                        Alist.pop(index)
                        prelist.pop(index)

        Zp = list(zip(prelist, Alist))

        return Zp

    
    def euler(self, a, A):
        
        I = (0,)*len(A)
        Xp = self.prepare(np.cos(a), I)
        Yp = self.prepare(1j*np.sin(a), A)

        return Xp + Yp 

    
    def KXK(self, a, K_r, Xp):

        b = a if type(a) == list else [a]
        K = K_r if type(K_r) == list else [K_r]
        Zp = Xp

        for i in range(len(K)):
        
            Kp = self.euler(b[-(1+i)], K[-(1+i)])
            Kdagp = self.euler(-b[-(1+i)], K[-(1+i)])
            Yp = self.fullprod(Kp, Zp)
            Zp = self.fullprod(Yp, Kdagp)

        return Zp 

    
    def trace(self, X):
        t = 0
    
        for i in range(len(X)):
    
            if X[i][1] == (0,)*len(X[i][1]):
                t += X[i][0]
    
        return t



class Hamiltonians:
    
    
    def __init__(self, N, model = None):
        self.N = N
        self.model = model
        
        
    def Hamiltonian(self, coeff = None):
    
        if self.model == 'TFIM':
            H = []
            p = []
            x = [1]*self.N if coeff == None else coeff
            

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                p.append(1)

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))
                p.append(x[i])

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))
            p.append(x[-1])

            return H, p
        
           
        elif self.model == 'XY':
            H = []
            p = [1]*2*(self.N-1)
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

            return H, p
        
        
        elif self.model == 'TFXY':
            H = []
            p = []
            x = [1]*self.N if coeff == None else coeff

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                p.append(1)
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))
                p.append(1)

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))
                p.append(x[i])

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))
            p.append(x[-1])

            return H, p
        
        
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
            p = [1]*3*(self.N-1)
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
                
            return H, p
        
        
        elif self.model == 'CFXY':
            H = []
            p = []
            x = [(1,1)]*self.N if coeff == None else coeff

            for i in range(self.N-1):

                l = [0]*self.N
                l[i] = 1
                l[i+1] = 1
                H.append(tuple(l))
                p.append(1)
                
                l = [0]*self.N
                l[i] = 2
                l[i+1] = 2
                H.append(tuple(l))
                p.append(1)

                l = [0]*self.N
                l[i] = 3
                H.append(tuple(l))
                p.append(x[i][0])
                
                l = [0]*self.N
                l[i] = 2
                H.append(tuple(l))
                p.append(x[i][1])

            l = [0]*self.N
            l[self.N-1] = 3
            H.append(tuple(l))
            p.append(x[-1][0])
            
            l = [0]*self.N
            l[self.N-1] = 2
            H.append(tuple(l))
            p.append(x[-1][1])

            return H, p
   

        else:
            
            return []
    
    
    def algebra(self):
        
        g = self.Hamiltonian()[0]
        s = [-1]*len(g)
        finalindex = len(g) - 1
        initialindex = -1
        t = True
        cont = False
        
        while t == True:
            t = False
            
            for i in range(finalindex, initialindex, -1):
                for j in range(i-1, -1, -1):
                    Com = Pauli().commutator(g[i],g[j])
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
        


class Cartan(Hamiltonians):


    def __init__(self, N, model = None):
        Hamiltonians.__init__(self, N, model)

    
    
    def decomposition(self):
        
        A = self.algebra()[0]
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
        
        for i in range(1,len(m_sorted)):
            for j in range(len(h)):
                
                if Pauli().commutator(m_sorted[i],h[j]) != 0:
                    break
                    
                elif j == len(h)-1:
                    h.append(m_sorted[i])
                    
        return h
 
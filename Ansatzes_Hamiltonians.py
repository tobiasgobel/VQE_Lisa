from pauli_objects import *
from Energy_funcions import *
def QAOA(N:int, L:int, array_method = False):
    ansatz = []
    assert L%2 == 0, "L is odd"

    XX_layer = [pauli(f"X{i}X{i+1}",N) for i in range(N-1)] # not yet flattened circuit
    Z_layer = [pauli(f"Z{i}", N) for i in range(N)]

    for i in range(int(L/2)):
        ansatz += XX_layer
        ansatz += Z_layer

    if array_method:
        ansatz = [a.to_parray() for a in ansatz]
    return ansatz

def random_circuit(N:int, L:int, array_method = False):
    ansatz = []
    assert L%2 == 0, "L is odd"

    for i in range(int(L/2)):
        sigmas = np.random.choice(["X","Y","Z"], (2,N-1))
        sigma_3 = np.random.choice(["X","Y","Z"], N)

        pair_layer = [pauli(f"{sigmas[0][i]}{i}{sigmas[1][i]}{i+1}",N) for i in range(N-1)]
        ansatz += pair_layer

        single_layer = [pauli(f"{sigma_3[i]}{i}", N) for i in range(N)]
        

        ansatz += single_layer

    if array_method:
        ansatz = [a.to_parray() for a in ansatz]
    return ansatz


def Z_expectation_val(N, array_method = False):
    first_Z = pauli(f"Z{0}", N)
    middle_Z = pauli(f"Z{int(N/2)}", N)

    H = [first_Z*middle_Z]
    if array_method:
        H = [h.to_parray() for h in H]
    return H

def TUCC(N:int, Layers:int,array_method=False):
    XY_layer = [pauli(f"X{i}Y{i+1}", N) for i in range(N-1)]
    ansatz = XY_layer*Layers
    if array_method:
        ansatz = [a.to_parray() for a in ansatz]
    return ansatz

def TFIM(N, X_h, Z_h=-1,array_method=False):
    Z_terms = [pauli(f'Z{i}',N,Z_h) for i in range(N)]
    X_terms = [pauli(f'X{i}X{i+1}',N,X_h) for i in range(N-1)]
    H = X_terms + Z_terms

    if array_method:
        H = [h.to_parray() for h in H]
    return H



### QAOA to Max E3LIN2 problem

def E3LIN2(N, D):

    #Randomly generate clauses
    clauses = []
    for i in range(D):
        a = np.random.choice([-1,1],3)
        b = np.random.choice(range(N),3,replace=False)
        clauses.append((a,b))

    #Hamiltonian

    H = []
    for a,b in clauses:
        H.append(pauli(f"Z{b[0]}Z{b[1]}Z{b[2]}", N, .5*a[0]*a[1]*a[2]))
    return H

def E3LIN2_ansatz(N, H, L = 1, array_method = False):
    #hadamard gates in terms of pauli matrices
    #Y rotation followed by X rotation
    ansatz = []
    ansatz_full = []

    #Build the ansatz
    for i in range(L):
        for j in range(N):
            ansatz_full.append(pauli(f"Y{j}", N))
        for h in H:
            ansatz.append(h.factor**-1*(h))
            ansatz_full.append(h.factor**-1*(h))
        for n in range(N):
            ansatz_full.append(pauli(f"X{n}", N))


    return ansatz, ansatz_full

def matchgate_hamiltonian(N, z_h = 1, zz_h = 1, xx_h = 1):
    H = []
    for i in range(N):
        H.append(pauli(f"Z{i}",N,z_h))
        if i == N-1:
            break
        H.append(pauli(f"Z{i}Z{i+1}",N,zz_h))
        H.append(pauli(f"X{i}X{i+1}",N,xx_h))
    return H

def matchgate_ansatz(N, K, array_method = False, ZZ_gates = True):

    XX_layer = [pauli(f"X{i}X{i+1}",N) for i in range(N-1)] # not yet flattened circuit
    ZZ_layer = [pauli(f"Z{i}Z{i+1}",N) for i in range(N-1)] 
    Z_layer = [pauli(f"Z{i}", N) for i in range(N)]

    for i in range(int(K)):
        ansatz += XX_layer
        ansatz += Z_layer
        if ZZ_gates:
            ansatz += ZZ_layer

    if array_method:
        ansatz = [a.to_parray() for a in ansatz]
    return ansatz






    


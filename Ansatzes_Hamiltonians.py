from pauli_objects import *

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
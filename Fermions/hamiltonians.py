from objects import *

def fermionic_TFIM(n, j=-1, h=-1):
    H = []
    for i in range(n-1):
        H.append(monomial(n, [2*i, 2*(i+1)+1], 1j*h))
    for i in range(n):
        H.append(monomial(n, [2*i, 2*i+1], 1j*j))
    return H


def ansatz_TFIM(n):
    H = []
    counter = 0
    for i in range(n-1):
        H.append(monomial(n, [2*i, 2*(i+1)+1], 1, counter))
        counter+=1
    for i in range(n):
        H.append(monomial(n, [2*i, 2*i+1], 1,counter))
        counter+=1
    return H
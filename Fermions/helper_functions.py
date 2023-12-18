from objects import *

def Pfaffian(matrix):
    if matrix.shape == (2,2):
        return matrix[0,1]
    else:
        return np.sqrt(np.abs(np.linalg.det(matrix)))**4

def commute(monomial1,monomial2):
    assert monomial1.n == monomial2.n
    len_1 = len(monomial1.positions)
    len_2 = len(monomial2.positions)
    overlap = len(set(monomial2.positions).intersection(set(monomial1.positions)))
    if (-1)**(len_1*len_2 + overlap)==1:
        return True
    else:
        return False


def lightcone(monomial, circuit, gaussian=False, reverse = False):
    lc = []
    indices = []
    circuit = circuit if not reverse else circuit[::-1]
    
    for i, m in enumerate(circuit):
        #skip non-gaussian gates if gaussian lightcone
        if gaussian and len(m)==4:
            continue
        i = i if not reverse else len(circuit)-i-1

        #check if the monomial commutes with the observable
        if not commute(monomial, m):
            lc.append(m)
            indices.append(i)
            continue

        #check if it commutes with other gates
        for l in lc:
            if not commute(monomial, l):
                lc.append(l)
                indices.append(i)
                break
    return lc, indices
        








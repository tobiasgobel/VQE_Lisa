from objects import *
from hamiltonians import *
from helper_functions import *
def calc_state(monomial, x):
    #even parity
    for majorana in monomial.positions[::-1]:
        fermion_index = int(majorana/2) if majorana % 2 == 0 else int((majorana-1)/2)

        sign = (-1)**(x.x[:fermion_index].count(1))
        if majorana % 2 == 0:
            #c_k = i(a_k + a_k^dagger)= zz...z_{k-1}x_k
            x = sign*x.xflip(fermion_index)

        elif majorana % 2 == 1:
            #c_k = (a_k + a_k^dagger)= zz...z_{k-1}y_k
            x = sign*x.yflip(fermion_index)
    return x*monomial.factor

def zero_corr(n):
    corr_0 = np.zeros((2*n,2*n), dtype=np.complex128)
    for i in range(n):
        corr_0[2*i,2*i+1] = 1
        corr_0[2*i+1,2*i] = -1
    return corr_0


def calc_correlation_matrix(n, ansatz, angles):
    corr_0 = zero_corr(n)
    circuit = [GaussianRotation(ansatz[i], angles[i]) for i in range(len(ansatz))]
    O = np.eye(2*n, dtype=np.complex128)
    for i in circuit:
        O = O* i.compute_matrix_repr()
    corr = O.T @ corr_0 @ O
    return corr

def energy(n, H, ansatz, angles):
    corr = calc_correlation_matrix(n, ansatz, angles)
    print(corr)
    E = 0
    for h in H:
        #select the submatrix of the correlation matrix
        sub_corr = corr[h.positions, :][:, h.positions]
        print(sub_corr, h.factor)
        #add energy contribution
        E += -1j*h.factor*Pfaffian(sub_corr)
    return E

n = 3
H = fermionic_TFIM(n)
ansatz = ansatz_TFIM(n)
angles = [np.pi]*len(ansatz)
print(f"Hamiltonian: {H}")
print(f"Ansatz: {ansatz}")
print(energy(n, H, ansatz, angles))
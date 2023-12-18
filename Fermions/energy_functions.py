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
    corr_0 = 1j*np.eye(2*n, dtype=np.complex128)
    for i in range(n):
        corr_0[2*i,2*i+1] = 1j
    corr_0 = corr_0 - corr_0.T

    assert np.allclose(corr_0@corr_0, np.eye(2*n, dtype=np.complex128))
    return corr_0


def calc_correlation_matrix(n, ansatz, angles):
    corr_0 = zero_corr(n)
    circuit = [GaussianRotation(ansatz[i], angles[i]) for i in range(len(ansatz))]
    O = np.eye(2*n, dtype=np.complex128)
    for i in circuit[::-1]:
        O =  i.matrix_repr @ O
    corr = O.T @ corr_0 @ O
    return corr

def energy(n, H, ansatz, angles):
    corr = calc_correlation_matrix(n, ansatz, angles)
    E = 0
    for h in H:
        #select the submatrix of the correlation matrix
        sub_corr = corr[h.positions, :][:, h.positions]
        #add energy contribution
        term =  -1j*h.factor*Pfaffian(sub_corr)
        E += term
    return E

n = 10
H = fermionic_TFIM(n)
ansatz = ansatz_TFIM(n)
angles = np.linspace(0.3, 2*np.pi, len(ansatz))
print('angles', angles)
z_contribution = -1*(np.cos(angles[0])**2 - np.sin(angles[0])**2)
xx_contribution = -4*np.sin(angles[1])*np.cos(angles[1])*np.cos(angles[0])*np.sin(angles[0])
print('energy quantum', energy(n, H, ansatz, angles))
print('energy classical', 2* z_contribution+xx_contribution)



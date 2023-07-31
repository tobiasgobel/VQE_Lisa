
from K_cell_searching import *
from cirq_energy import *
from time import time
from visualize_landscape import *
from sys import getsizeof
N = 28
H = Z_expectation_val(N,array_method = False)
ansatz = random_circuit(N, 4, array_method=False)
#ansatz = QAOA(N, 4, array_method=False)
HVA = False
order = 4
print(f"Number of qubits: {N}")
print(f"Length of ansatz: {len(ansatz)}")
print(f"Order of approx: {order}")


Energy_mat = False
Energy_appr = True
Energy_cirq = True
Energy_appr_0 = True

thetas = np.random.choice([0,-np.pi/2, np.pi/2], len(ansatz))/2
K = np.round(thetas/np.pi*4, 0).astype(int)
K_0 = np.zeros(len(ansatz), dtype  = int)

delta_theta = np.random.randn(len(ansatz))
thetas_moved = thetas - K*np.pi/4


#Energy with matrix calculation
if Energy_mat:
    time_matrix  = time()
    matrix_ansatz = [t.matrix_repr() for t in ansatz]
    matrix_H = sum([h.matrix_repr() for h in H])
    E_matrix = Energy_matrix(thetas, N, matrix_H, matrix_ansatz, K_0)
    time_matrix = time() - time_matrix
    print(f"{'E_matrix:':<25} {f'{E_matrix}'}\n", f"{'time:':<25} {f'{time_matrix}'}\n")

#Energy with approximation method and perturbation
if Energy_appr:
    time_expansion = time()
    s = s_dict(N, ansatz, K, order)
    G_K = G_k(N, H, ansatz,K)
    E_expansion = energy(thetas, s, G_K, order)
    time_expansion = time() - time_expansion
    print(f"{'E_expansion:':<25} {f'{E_expansion}'}\n", f"{'time:':<25} {f'{time_expansion}'}\n")


#Energy with approximation method
if Energy_appr_0:
    time_expansion = time()
    s_0 = s_dict(N, ansatz, K_0, order)
    G_K_0 = G_k(N, H, ansatz,K_0)
    E_expansion_0 = energy(thetas, s_0, G_K_0, order)
    time_expansion_0 = time() - time_expansion
    print(f"{'E_expansion_0:':<25} {f'{E_expansion_0}'}\n", f"{'time:':<25} {f'{time_expansion_0}'}\n")

#Energy with cirq
if Energy_cirq:
    time_cirq = time()
    H_cirq = sum([h.cirq_repr() for h in H])
    ansatz_cirq = [a.cirq_repr() for a in ansatz]
    E_cirq = cirq_Energy(thetas, N, ansatz_cirq, H_cirq, K_0, HVA)
    time_cirq = time() - time_cirq
    print(f"{'E_cirq:':<25} {f'{E_cirq}'}\n", f"{'time:':<25} {f'{time_cirq}'}\n")








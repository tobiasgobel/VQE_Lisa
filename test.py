
from K_cell_searching import *
from cirq_energy import *
from time import time
N = 6
H = TFIM(N,1)
HVA = False
ansatz = QAOA(N, 2)

print(f"length of ansatz: {len(ansatz)}")
matrix_ansatz = [t.matrix_repr() for t in ansatz]
matrix_H = sum([h.matrix_repr() for h in H])


if HVA:
    thetas = (np.random.rand(HVA)-.5)*np.pi/4
    K = np.random.randint(0,3,HVA)-1
    K = distribute_over_gates(HVA, N, K)

else:
    thetas = (np.random.rand(len(ansatz))-.5)*np.pi/4
    K = np.random.randint(0,3,len(ansatz))-1


order = 2*len(ansatz)


time_matrix  = time()
#Energy with matrix calculation
E_matrix = Energy_matrix(thetas, N, matrix_H, matrix_ansatz, K)
time_matrix = time() - time_matrix

#Energy with approximation method
time_expansion = time()
s = s_dict(N, ansatz, K, order)
#s = s_dict(N, ansatz, K, order)
G_K = G_k(N, H, ansatz,K)
E_expansion = energy(thetas, s, G_K, order, HVA = HVA)
time_expansion = time() - time_expansion


#Energy with cirq
time_cirq = time()
H_cirq = sum([h.cirq_repr() for h in H])
ansatz_cirq = [a.cirq_repr() for a in ansatz]
E_cirq = cirq_Energy(thetas, N, ansatz_cirq, H_cirq, K)
time_cirq = time() - time_cirq


print(f"{'E_matrix:':<25} {f'{E_matrix}'}\n", f"{'time:':<25} {f'{time_matrix}'}\n")
print(f"{'E_expansion:':<25} {f'{E_expansion}'}\n", f"{'time:':<25} {f'{time_expansion}'}\n")
print(f"{'E_cirq:':<25} {f'{E_cirq}'}\n", f"{'time:':<25} {f'{time_cirq}'}\n")

#check if energies match
if np.abs(E_matrix - E_expansion)< 1e-3 and order == 2*len(ansatz):
    print("Energies match!")
elif order != 2*len(ansatz):
    print("Not the full order is used for the approximation")
else:
    print("Energies dont match")



# args = (s, G_K, order, HVA)
# opt = E_optimizer(energy, thetas, args = args, boundary = "hypersphere", epsilon= 1e-3)
# E = opt.optim()

# angles = E.x
# print("After optimization:", end = "\n\n")
# print("E_approximated:", E.fun)
# print("E_cirq:", cirq_Energy(angles, N, ansatz_cirq, H_cirq, K))
# minimized_cirq = scipy.optimize.minimize(cirq_Energy, thetas, args = (N, ansatz_cirq, H_cirq, K), method = "L-BFGS-B")
# print("E_cirq minimized:", minimized_cirq.fun)
# print("angles cirq minimized:", minimized_cirq.x)
# print("K:", K)

# res = find_K(N, ansatz, H, 10, order, log = False, matrix_min = None, boundary = "hypersphere", HVA = HVA)
# print(res)
# if np.abs(E1 - E2)< 1e-3 and order == 2*len(ansatz):
#     print("Energies match!")
# elif order != 2*len(ansatz):
#     print("Not the full order is used for the approximation")
# else:
#     print("Energies dont match")
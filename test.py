
from K_cell_searching import *
from cirq_energy import *
from time import time
from visualize_landscape import *
N = 10
H = TFIM(N,1)
HVA = False
ansatz = QAOA(N, 2)
Nfeval = 1
print(f"length of ansatz: {len(ansatz)}")
# matrix_ansatz = [t.matrix_repr() for t in ansatz]
# matrix_H = sum([h.matrix_repr() for h in H])

def callbackF(Xi, state):
    global Nfeval
    E = state.fun
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}  {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], E))
    Nfeval += 1


if HVA:
    thetas = (np.random.rand(HVA)-.5)*np.pi/4
    K = np.random.randint(0,3,HVA)-1
    K = distribute_over_gates(HVA, N, K)

else:
    thetas = (np.random.rand(len(ansatz))-.5)*np.pi/4
    K = np.random.randint(0,3,len(ansatz))-1


order = 4


time_matrix  = time()
#Energy with matrix calculation
E_matrix = 0#Energy_matrix(thetas, N, matrix_H, matrix_ansatz, K)
time_matrix = time() - time_matrix

#Energy with approximation method
time_expansion = time()
s = s_dict(N, ansatz, K, order)

#s = s_dict(N, ansatz, K, order)
G_K = G_k(N, H, ansatz,K)
E_expansion = energy(thetas, s, G_K, order, HVA = HVA)
time_expansion = time() - time_expansion
print(time_expansion)


#Energy with cirq
time_cirq = time()
H_cirq = sum([h.cirq_repr() for h in H])
ansatz_cirq = [a.cirq_repr() for a in ansatz]
E_cirq = cirq_Energy(thetas, N, ansatz_cirq, H_cirq, K)
time_cirq = time() - time_cirq


print(f"{'E_matrix:':<25} {f'{E_matrix}'}\n", f"{'time:':<25} {f'{time_matrix}'}\n")
print(f"{'E_expansion:':<25} {f'{E_expansion}'}\n", f"{'time:':<25} {f'{time_expansion}'}\n")
print(f"{'E_cirq:':<25} {f'{E_cirq}'}\n", f"{'time:':<25} {f'{time_cirq}'}\n")



# #check if energies match
# if np.abs(E_matrix - E_expansion)< 1e-3 and order == 2*len(ansatz):
#     print("Energies match!")
# elif order != 2*len(ansatz):
#     print("Not the full order is used for the approximation")
# else:
#     print("Energies dont match")




# E = scipy.optimize.minimize(Energy_matrix, thetas, args = (N, matrix_H, matrix_ansatz, K))

# angles = E.x

# minimized_cirq = scipy.optimize.minimize(cirq_Energy, thetas, args = (N, ansatz_cirq, H_cirq, K))

# # 
# angles = minimized_cirq.x
# print(E.fun, minimized_cirq.fun)

# args = (N, ansatz_cirq, H_cirq, K)
# args2 = (N, matrix_H, matrix_ansatz, K)
# landscape_visualize(angles, cirq_Energy, args, scale = np.pi ,num_directions=len(ansatz), filename = "cirq-landscape.png")
# landscape_visualize(angles, Energy_matrix, args2, scale = np.pi , num_directions = len(ansatz), filename = "matrix-landscape.png")

# print("E_cirq minimized:", minimized_cirq)
# print("angles cirq minimized:", minimized_cirq.x)
# print("K:", K)



# res = find_K(N, ansatz, H, 10, order, log = False, matrix_min = None, boundary = "hypersphere", HVA = HVA)
# print(res[:-1])
# if np.abs(E1 - E2)< 1e-3 and order == 2*len(ansatz):
#     print("Energies match!")
# elif order != 2*len(ansatz):
#     print("Not the full order is used for the approximation")
# else:
#     print("Energies dont match")
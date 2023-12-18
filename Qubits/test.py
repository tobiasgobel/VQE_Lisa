
from K_cell_searching import *
from cirq_energy import *
from time import time
from visualize_landscape import *
from sys import getsizeof
N = 10


H = TFIM(N,X_h= -1)
HVA = False
ansatz = QAOA(N, 2)

# print("H: ", H)
# print("ansatz: ",ansatz)
order = 4
Nfeval = 1
print(f"Number of qubits: {N}")
print(f"Length of ansatz: {len(ansatz)}")
print(f"Order of approx: {order}")
# matrix_ansatz = [t.matrix_repr() for t in ansatz]
# matrix_H = sum([h.matrix_repr() for h in H])


if HVA:
    # thetas = (np.random.rand(HVA)-.5)*np.pi/4
    # K = np.random.randint(0,3,HVA)-1

    thetas = np.ones(HVA)
    K = np.zeros(HVA,dtype = int)
    K = distribute_over_gates(HVA, N, K)

else:
    thetas_torch =torch.tensor((np.random.rand(len(ansatz))-.5)*np.pi/4, requires_grad=True)
    thetas = thetas_torch.detach().numpy()
    K = np.random.randint(0,3,len(ansatz))-1
    K = np.zeros(len(ansatz), dtype = int)
    thetas = np.ones(len(ansatz))*np.pi
    thetas = np.linspace(0.3, 2*np.pi, len(ansatz))
    # print("K-cells: ", K)



time_matrix  = time()
#Energy with matrix calculation
E_matrix = 0#Energy_matrix(thetas, N, matrix_H, matrix_ansatz, K)
time_matrix = time() - time_matrix

#with lightcones
time_expansion_lc = time()
sdicts, lc = s_dicts_lightcones(N, ansatz, H, K, order)
time_sdicts = time() - time_expansion_lc
G_K = G_k(N, H, ansatz,K)
E_expansion_lc = energy_lightcone(thetas, sdicts, G_K, lc, order)
time_expansion_lc = time() - time_expansion_lc
print(time_expansion_lc, time_sdicts)




#Energy with cirq
time_cirq = time()
H_cirq = sum([h.cirq_repr() for h in H])
ansatz_cirq = [a.cirq_repr() for a in ansatz]
E_cirq = cirq_Energy(thetas, N, ansatz_cirq, H_cirq, K, HVA)
time_cirq = time() - time_cirq


#Energy with approximation method
time_expansion = time()
# print('s',s)
# print(f"memory s-dict: {getsizeof(s)}")
s = 1#s_dict(N, ansatz, K, order)
G_K = G_k(N, H, ansatz,K)

E_expansion = 0#energy(thetas, s, G_K, order, HVA = HVA)
time_expansion = time() - time_expansion
print(time_expansion)

print(f"{'E_matrix:':<25} {f'{E_matrix}'}\n", f"{'time:':<25} {f'{time_matrix}'}\n")
print(f"{'E_expansion:':<25} {f'{E_expansion}'}\n", f"{'time:':<25} {f'{time_expansion}'}\n")
print(f"{'E_cirq:':<25} {f'{E_cirq}'}\n", f"{'time:':<25} {f'{time_cirq}'}\n")
print(f"{'E_expansion_lc:':<25} {f'{E_expansion_lc}'}\n", f"{'time:':<25} {f'{time_expansion_lc}'}\n")


#check if energies match
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

#print("E_cirq minimized:", minimized_cirq)
# print("angles cirq minimized:", minimized_cirq.x)
# print("K:", K)
# resultt = gradient_optimizer(thetas_torch, energy, args = (s,G_K, 10,False,True),max_iter = 200)
# print(f"angles: {resultt.x}")
# print(f"Energy: {resultt.fun}")
# print(f"iters: {resultt.nfev}")

# res = find_K(N, ansatz, H, 10, order, log = False, matrix_min = None, boundary = "hypersphere", HVA = HVA)
# print(res[:-1])
# if np.abs(E1 - E2)< 1e-3 and order == 2*len(ansatz):
#     print("Energies match!")
# elif order != 2*len(ansatz):
#     print("Not the full order is used for the approximation")
# else:
#     print("Energies dont match")

from K_cell_searching import *
from cirq_energy import *
from time import time
from visualize_landscape import *
from sys import getsizeof
N = 20

H = Z_expectation_val(N,array_method = False)
ansatz_full = random_circuit(N, 8, array_method=False)
lightc = lightcone(H, ansatz_full, order_max = 12)
for i in lightc: print(len(lightc[i]))
ansatz = lightc[1]
print(len(ansatz), len(ansatz_full))


#ansatz = QAOA(N, 2, array_method=False)
HVA = False
order = 2*len(ansatz)

print("------------------")
print(f"Number of qubits: {N}")
print(f"Length of ansatz: {len(ansatz)}")
print(f"Order of approx: {order}")
print("------------------")


Energy_mat = False
Energy_appr = True
Energy_cirq = True
Energy_appr_0 = False


def reproduce(N, H, ansatz, order, thetas, delta_theta, Energy_mat = False, Energy_appr = True, Energy_cirq = True, Energy_appr_0 = False):
    #Generate random thetas and KA

    thetas = thetas + delta_theta
    K = np.zeros(len(ansatz), dtype  = int) #np.array([0]*N+[2]*N+[0]*D+[0]*N)#np.round(thetas/np.pi*4, 0).astype(int)
    thetas_moved = thetas - K*np.pi/4
    K_0 = np.zeros(len(ansatz), dtype  = int) 

    E_matrix = None
    E_expansion = None
    E_cirq = None
    E_expansion_0 = None


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
        s = s_dict(N, ansatz, K, order, prepare_x_basis = True)
        #Cliffords x gates
        cliffords = [pauli(f'X{i}', N, -1) for i in range(N)]
        G_K = G_k(N, H, cliffords, [-1]*len(cliffords))
        G_K = G_k(N, G_K, ansatz, K)
        #cliffords of y rotations
        G_K = G_K_prepare_x_basis(N, G_k(N, H, ansatz, K))

        E_expansion = energy(thetas_moved, s, G_K, order)
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

    return E_matrix, E_expansion, E_cirq, E_expansion_0


# for delta_theta in deltas:
#     First_time = True
#     for order in orders:
#             print(f"delta_theta: {delta_theta} and order: {order}")
#             try:
#                 if not First_time:
#                     _, result, _ , _  = reproduce(N, H, ansatz, order, thetas, delta_theta, Energy_cirq = False)
#                     data[delta_theta][order] = (result, result_cirq)
#                 elif First_time:
#                     _,result, result_cirq, _ = reproduce(N, H, ansatz, order, thetas, delta_theta)
#                     data[delta_theta][order] = (result, result_cirq)
#             except:
#                 print("error")
#                 data[delta_theta][order] = None
#             First_time = False
# data.to_csv("reproducing_results/data/repr_3.csv", sep = ";")

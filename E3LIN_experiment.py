from K_cell_searching import *
from cirq_energy import *
from time import time
from visualize_landscape import *
from sys import getsizeof
N = 50
D = 4
H = E3LIN2(N, D)
ansatz, ansatz_full = E3LIN2_ansatz(N, H, 1)


# cliffords over observables
cliffords = [pauli(f'X{i}',N,-1) for i in range(N)]
H = G_k(N, H, cliffords, [1]*N)

#lightcone 
LightCone = lightcone(H, ansatz, order_max = 12)
for i in H: print(i)
print(LightCone)
LC_order = 3
ansatz = LightCone[LC_order]
ansatz = ansatz[::-1]

#G_K
K = np.zeros(len(ansatz), dtype =int)
G_K = G_k(N, H, ansatz, K)

#s_dict
order = 2*len(ansatz)   
s = s_dict(N, ansatz, K, 2*len(ansatz), prepare_x_basis = True)





gammas = np.arange(0, np.pi/2, .04)
energies_appr = []
energies_exact = []


for gamma in gammas:
    thetas = np.array([gamma]*len(ansatz))
    
    print(f"gamma: {gamma}")

    #E appr
    order = 10
    E_appr = energy(thetas, s, G_K, order)
    energies_appr.append(E_appr)

    #E cirq
    thetas_full = [-np.pi/4]*N + [gamma]*D + [-np.pi/4]*N
    time_cirq = time()
    H_cirq = sum([h.cirq_repr() for h in H])
    ansatz_cirq = [a.cirq_repr() for a in ansatz_full]
    E_cirq = cirq_Energy(thetas_full, N, ansatz_cirq, H_cirq, K)
    time_cirq = time() - time_cirq
    print(f"{'E_cirq:':<25} {f'{E_cirq}'}\n", f"{'time:':<25} {f'{time_cirq}'}\n")
    energies_exact.append(E_cirq)





plt.plot(gammas, energies_appr, label = "approx")
plt.plot(gammas, energies_exact, label = "exact")
plt.legend()
plt.savefig("reproducing_results/reproduced_graphs/energy_gamma.png")


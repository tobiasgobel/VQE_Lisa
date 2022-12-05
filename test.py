
from VQE_functions import *

N = 4
H = TFIM(N,-1)
ansatz = TUCC(N, 2)

print(f"length of ansatz: {len(ansatz)}")
matrix_ansatz = [t.matrix_repr() for t in ansatz]
matrix_H = sum([h.matrix_repr() for h in H])

thetas = (np.random.rand(len(ansatz))-.5)*np.pi/4

K = np.random.randint(0,3,len(ansatz))-1
order = 6#len(ansatz)*2
print(f"K= {K}\n")
print(f"theta = {thetas} \n")


#Energy with matrix calculation
E1 = Energy_matrix(thetas, N, matrix_H, matrix_ansatz, K)

#Energy with approximation method
s = s_dict2(N, ansatz, K, order)
s = s_dict(N, ansatz, K, order)
G_K = G_k(N, H, ansatz,K)
E2 = energy(thetas, s, G_K, order)
print(E1, E2)

args = (s, G_K, order)
opt = E_optimizer(energy, thetas, args = args, boundary = "hypersphere", epsilon= 1e-3)
E = opt.optim()
print(E)

print(find_K(N, ansatz, H, 10, order))
if np.abs(E1 - E2)< 1e-3 and order == 2*len(ansatz):
    print("Energies match!")
elif order != 2*len(ansatz):
    print("Not the full order is used for the approximation")
else:
    print("Energies dont match")
 
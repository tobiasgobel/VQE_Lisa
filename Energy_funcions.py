from Func import *

def G_k(N, H, ansatz, K):
  g_k = []

  #Initialize list of Clifford gates with respective power of K.
  G_K = []
  for i in range(len(K)):
    G_K += [np.sign(K[i])*ansatz[i]]*abs(K[i])
  for P in H:
    # G_K = [ansatz[i]**K[i] for i in range(len(K))]
    #Apply nested Clifford Map to obtain G^-K P_a G^K
    paulistring = reduce(Clifford_map, [P]+G_K[::-1])
    g_k += [paulistring]
  return g_k



def power_product(x,y):
    out = (x[0]**y[0]).copy()
    for i in range(1,len(x)):
        if y[i]==1:
          out.multiply(x[i])
    return out





@jit(nopython=True)
def dict_multiplication(k,values,thetas):
    sum = 0
    for i in range(k.shape[0]):
        product = 1
        for j in range(k.shape[1]):
            product*=(np.sin(thetas[j]))**k[i,j]*(np.cos(thetas[j]))**(-k[i,j]+1)
        sum += product*values[i]
    return sum

def Normalize(s_d, thetas, order):
    sum = 0
    for s in s_d:
        k, values = s_d[s]
        k1, values1 = s_d[s]
        factor = dict_multiplication(k,values,thetas)
        factor1 = dict_multiplication(k1,values1,thetas)
        sum += np.conj(factor1)*factor

    return sum

def energy(thetas, s_dict,G_K, order, HVA=False):
  N = len(list(s_dict.keys())[0])
  E = 0
  s_dict1 = s_dict

  if HVA:
    thetas = distribute_over_gates(HVA, N, thetas)

  for paulistring in G_K: #loop through terms in Hamiltonian
    E_a = 0
    #loop over basis states
    for s in s_dict1:
      E_a_s = 0
    
      #Calculate G^-K P_a G^K |s>
      a, state = paulistring.state(s)
      #Define contributions of |s> and |s'>
      psi_s1 = s_dict1[s]

      #Check if the state created by hamiltonian, exists in wavefunction
      try:
        psi_s2 = s_dict1[state]
      except:
        break

      A = dict_multiplication(psi_s1[0],psi_s1[1],thetas)
      B = dict_multiplication(psi_s2[0],psi_s2[1],thetas)
      E_a_s = A*np.conj(B)

      E_a_s *= a
      E_a += E_a_s
    E += E_a
  
  norm = Normalize(s_dict1, thetas, order)
#   print(f"E:{E}, Norm{norm}, E_final:{E/norm}")
  return np.real(E/norm)

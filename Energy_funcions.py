from Func import *
import torch

def G_k(N, H, ansatz, K):
  g_k = []

  #Initialize list of Clifford gates with respective power of K.
  G_K = []
  for i in range(len(K)):
    G_K += [np.sign(K[i])*ansatz[i]]*abs(K[i])
  for P in H:
    #Apply nested Clifford Map to obtain G^-K P_a G^K
    paulistring = reduce(Clifford_map, [P]+G_K[::-1])
    g_k += [paulistring]
  return g_k

# @jit(nopython=True)
def dict_multiplication(k,values,thetas):
    sum = 0
    for i in range(k.shape[0]):
        product = 1
        for j in range(k.shape[1]):
          if k[i,j] == 1:
            product*= thetas[j]
        sum += product*values[i]
    return sum

def Normalize(terms):
    sum = 0
    for term in terms.values():
      sum+= term*torch.conj(term)
    return sum


def energy(thetas, s_dict,G_K, order, HVA=False, Pytorch= True):
  N = len(list(s_dict.keys())[0])
  E = 0
  s_dict1 = s_dict
  terms = {}
  thetas = np.tan(thetas) if Pytorch == False else torch.tan(thetas) #convert angles to tangents 

  if HVA:
    thetas = distribute_over_gates(HVA, N, thetas)

  for paulistring in G_K: #loop through terms in Hamiltonian
    E_a = 0
    #loop over basis states
    for s in s_dict1:
      #Calculate G^-K P_a G^K |s>
      a, state = paulistring.state(s)
      #Define contributions of |s> and |s'>
      psi_s1 = s_dict1[s]

      #Check if the state created by hamiltonian, exists in wavefunction
      try:
        psi_s2 = s_dict1[state]
      except:
        break


      
      if s not in terms:
        A = dict_multiplication(psi_s1[0],psi_s1[1],thetas)
        terms[s] = A
      else:
        A = terms[s]
      
      
      if state not in terms:
        B = dict_multiplication(psi_s2[0],psi_s2[1],thetas)
        terms[state] = B
      else:
        B = terms[state]


      E_a += a*A*torch.conj(B)
    E += E_a
  
  norm = Normalize(terms)
  return np.real(E/norm)

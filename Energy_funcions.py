from Func import *
import torch
from pauli_objects import *

def G_K_prepare_x_basis(N, G_K):
  #move y cliffords through T_K
  Y_rots = [pauli(f'Y{i}', N) for i in range(N)]
  K = [-1]*N
  G_K = G_k(N, G_K, Y_rots, K)
  return G_K

def G_K_add_cliffords(N, G_K, cliffords):
  #move y cliffords through T_K
  G_K = G_k(N, G_K, cliffords, [1]*len(cliffords))
  return G_K

def T_K_prepare_x_basis(N, T_K):
  #move y cliffords through T_K
  Y_rots = [pauli(f'Y{i}', N) for i in range(N)]
  T_K_new = []
  for P in T_K:
    for Y in Y_rots:
      P = Clifford_map(P,-1*Y)
    T_K_new += [P]
  return T_K_new


    

def lightcone(H, ansatz, order_max = 100):
    #H is a list of pauli objects
    #ansatz is a list of pauli objects

    #find the lightcone of the ansatz
    lightcone = []

    #initialize tree
    tree = H

    #loop through gates backwards
    for i, a in enumerate(ansatz[::-1]):

      #loop through gates in current tree
      for h in tree:
          R = a*h
          L = h*a

          #if ansatz gate does not commute with gate in tree
          if not R == L:
            #add index gate to lichtcone
            lightcone.append(len(ansatz)-i-1)
            #add gate to tree
            tree.append(a)
            tree.append(R)
            break

    return lightcone



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

@jit(nopython=True)
def dict_multiplication(k,values,thetas):
    sum = 0
    for i in range(k.shape[0]):
        product = 1
        for j in range(k.shape[1]):
          if k[i,j] == 1:
            product*= np.tan(thetas[j])
        sum += product*values[i]
    return sum

def Normalize(terms):
    sum = 0
    for term in terms.values():
      sum+= term*np.conj(term)
    return sum


def energy(thetas, s_dict,G_K, order = None, HVA=False, lightcone = None):
  N = len(list(s_dict.keys())[0])
  E = 0
  s_dict1 = s_dict
  terms = {}
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


      E_a += a*A*np.conj(B)
    E += E_a
  
  norm = Normalize(terms)
  return np.real(E/norm)



def energy_lightcone(thetas_full, s_dicts, G_K, lightcones,order = None):
  N = len(list(s_dicts[0].keys())[0])
  E = 0
  for i, paulistring in enumerate(G_K): #loop through terms in Hamiltonian
    terms = {}
    E_a = 0
    thetas = [thetas_full[j] for j in lightcones[i]]
    #loop over basis states
    s_dict1 = s_dicts[i]
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


      E_a += a*A*np.conj(B)
    print(terms)
    norm = 1 if terms == {} else Normalize(terms)
    E += E_a/norm
  
  return np.real(E)


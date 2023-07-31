
import time
from scipy.linalg import expm
from functools import *
from operator import *
import scipy
import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
from itertools import combinations
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r  took: %2.4f sec' %(f.__name__, te-ts))
        return result
    return wrap


@jit(nopython=True)
def pauli_on_pauli(p1,p2):
    if p1 == 'I':
        return 1, p2
    elif p2 == 'I':
        return 1, p1
    elif p1 == 'X' and p2 == 'Y':
        return 1j, 'Z'
    elif p1 == 'X' and p2 == 'X':
        return 1, 'I'
    elif p1 == 'Y' and p2 == 'Y':
        return 1, 'I'
    elif p1 == 'Z' and p2 == 'Z':
        return 1, 'I'
    elif p1 == 'Z' and p2 == 'X':
        return 1j, 'Y'
    elif p1 == 'Z' and p2 == 'Y':
        return -1j, 'X'
    else:
        a, p = pauli_on_pauli(p2,p1)
        return -1*a, p

@jit(nopython=True)
def single_pauli_action(pauli, spin):
    
    if pauli=='X':
        return((spin+1)%2, 1)
    elif pauli=='Y':
        return((spin+1)%2, 1j*(-1)**spin)
    elif pauli=='Z':
        return(spin, (-1)**spin)
    elif pauli=='I':
        return(spin, 1)
    else:
        print('wrong pauli!')
        return(None)

def distribute_over_gates(L, N, vector):
    out = []
    assert len(vector) == L
    for i in range(L):
        if i%2 == 0:
            out += [vector[i]]*(N-1)
        else:
            out += [vector[i]]*N
    assert len(out) == L*(N-1+N)/2
    return np.array(out)

def shorten_vector(L,N, vector):
    out = []
    for i in range(int(L/2)):
        out += [vector[i*(2*N-1)]]
        out += [vector[i*(2*N-1)+N]]
    return np.array(out)

@jit(nopython=True)
def Energy_eigen(H):
  result = np.linalg.eig(H)
  index = np.argmin(result[0])
  return result[0][index],result[1][index]


def Energy_matrix(thetas,N,H,ansatz, K, HVA = False):
    #build psi
    if HVA:
        thetas = distribute_over_gates(HVA, N, thetas)

    a = np.eye(2**N)
    zero_state = np.zeros(2**N)
    zero_state[0]=1
    for i in range(len(ansatz)-1,-1,-1):
        T = ansatz[i]
        exp = expm(1j*(np.pi/4*K[i]+thetas[i])*T)
        a = a @ exp
    psi = a @ zero_state
    #build Hamiltonian
    Energy = (np.transpose((np.conj(psi)) @ (H @ (psi))))

    return np.real(Energy)

def psi(thetas, ansatz,K, HVA = False):
    #build psi
    N = ansatz[0].shape[0].bit_length()-1
    if HVA:
        thetas = distribute_over_gates(HVA, N, thetas)
    a = np.eye(2**N)
    zero_state = np.zeros(2**N)
    zero_state[0]=1
    for i in range(len(ansatz)-1,-1,-1):
        T = ansatz[i]
        exp = expm(1j*(np.pi/4*K[i]+thetas[i])*T)
        a = a @ exp
    psi = a @ zero_state

    return psi


#gives result of transformation exp(-i*T1)*T2*exp(i*T2)
def Clifford_map(T1, T2, reversed_arguments = True):
  T1T2 = T1*T2
  T2T1 = T2*T1
  if T1T2.factor == T2T1.factor:
    if reversed_arguments:
      return T1
    else:
      return T2
  elif T1T2.factor == -T2T1.factor:
    if reversed_arguments:
      return -1j*T2T1
    else:
      return -1j*T1T2
  else:
    return "something wrong here"


#returns list of pauli objects that are the result 
#of pulling all clifford gates to the left
def pull_cliffords_through(ansatz, K, N):
  T_K = [ansatz[0]] 
  
  for j in range(1, len(ansatz)):
    T = ansatz[j]
    for i in range(j-1,-1,-1):
      for _ in range(abs(K[i])):
        T = Clifford_map(T,np.sign(K[i])*ansatz[i])
    T_K += [T] 
  return T_K



def angle_compare(theta_opt, theta_appr, K):
  theta_appr = theta_appr + np.array(K)*np.pi/4
  theta_appr = theta_appr % (2*np.pi)
  theta_opt = theta_opt % (2*np.pi)
  distance = np.linalg.norm(theta_opt-theta_appr)
  return distance

def wavefunction_compare(theta_opt, theta_appr, K, ansatz):
    wave_1 = psi(theta_opt, ansatz, [0]*len(ansatz))
    wave_2 = psi(theta_appr, ansatz, K)

    #       print(f"wavefunction exact: {wave_2}, wavefunction_approx: {wave_1}")
    overlap = np.abs(np.conj(wave_1) @ wave_2)
    return overlap

def overlap(theta_t, theta_a, K_init, K_a, ansatz, HVA=False):
    wave_1 = psi(theta_t, ansatz, K_init, HVA=HVA)
    wave_2 = psi(theta_a, ansatz, K_a, HVA=HVA)
    
    def overlap_phase(theta):
        return np.abs(np.conj(wave_1) @ (np.exp(1j*theta)*wave_2))
    overlap = scipy.optimize.minimize(overlap_phase, 0)
    return overlap.fun

def local_to_global_angle(thetas, K):
    thetas = thetas + np.array(K)*np.pi/4
    thetas = thetas % (2*np.pi)
    return thetas
    
def global_to_local_angle(thetas, K):
    thetas = thetas - np.array(K)*np.pi/4
    thetas = thetas % (2*np.pi)
    return thetas


def str_to_lst(N,string):
    pauli_lst = []
    pos_lst = []
    prev_int = False
    for k in string:
        if k.isdigit():
            if not prev_int:
                pos_lst.append(k)
            else:
                pos_lst[-1] += k
            prev_int = True
        else:
            pauli_lst.append(k)
            prev_int = False
    
    lst = []
    for n in range(N):
        if str(n) in pos_lst:
            index = pos_lst.index(str(n))
            lst+=pauli_lst[index]
        else:
            lst+="I"
    return lst


def HVA_initialisation(HVA, max_angle):
    thetas = []
    linspace = np.linspace(1e-4, max_angle, HVA//2)
    for gate in range(HVA):
        if gate%2 == 1:
            thetas.append(max_angle)
        elif gate%2 == 0:
            thetas.append(linspace[int(gate/2)])
    return thetas

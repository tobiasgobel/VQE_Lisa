
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


def power_product(x,y):
    out = (x[0]**y[0]).copy()
    for i in range(1,len(x)):
        if y[i]==1:
          out.multiply(x[i])
    return out


@timing
@jit(nopython=True)
def Energy_eigen(H):
  result = np.linalg.eig(H)
  index = np.argmin(result[0])
  return result[0][index],result[1][index]

def Energy_matrix(thetas,N,H,ansatz, K):
  #build psi
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

def psi(thetas, ansatz,K):
    #build psi
  N = ansatz[0].shape[0].bit_length()-1
  a = np.eye(2**N)
  zero_state = np.zeros(2**N)
  zero_state[0]=1
  for i in range(len(ansatz)-1,-1,-1):
    T = ansatz[i]
    exp = expm(1j*(np.pi/4*K[i]+thetas[i])*T)
    a = a @ exp
  

  psi = a @ zero_state
  
  return psi

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


Pauli = {"X":np.array([[0,1],[1,0]]), "Z": np.array([[1,0],[0,-1]]),"Y":np.array([[0,-1j],[1j,0]]), "I":np.eye(2)}

class pauli:
  def __init__(self,string, N, factor = 1):
    if isinstance(string, list):
        self.string = string
    elif isinstance(string, str):
        self.string = str_to_lst(N, string)

    self.factor = factor
    self.N = N
    self.starting_state = np.array([0]*self.N)


  def __str__(self):
    return f"{self.string}   factor: "+str(self.factor)

  #define multiplying by a constant (on left hand side)
  def __rmul__(self, c):
    return pauli(self.string,self.N, c*self.factor)

  #define the power of a pauli string
  def __pow__(self, c):
    if c == 0:
        return pauli("I0",self.N)
    elif c == 1:
        return self
    else:
        C = pauli("I0",self.N)
        for i in range(abs(c)):
            C = C*self
        return C

  #define multiplying two pauli strings
  def __mul__(self, x):
      factor = self.factor*x.factor
      lst = [pauli_on_pauli(self.string[n],x.string[n]) for n in range(self.N)]
      factor = factor*np.prod([i for i,_ in lst])
      lst = [j for _,j in lst]
      return pauli(lst, self.N, factor)

  def multiply(self, x):
      lst = [pauli_on_pauli(self.string[n],x.string[n]) for n in range(self.N)]
      self.string = [j for _,j in lst]
      self.factor *= x.factor*np.prod([i for i,_ in lst])

  def copy(self):
      return pauli(self.string, self.N, self.factor)

    
  #calculate resulting state of paulistring when acted upon initial_state  
  def state(self, initial_state = 0):
    init_state = self.starting_state + initial_state
    a = self.factor
    for j, Pauli in enumerate(self.string):
      if Pauli =="I":
          continue
      spin = init_state[j]
      new_spin, factor = single_pauli_action(Pauli,spin)
      init_state[j] = new_spin
      a *= factor
    return a, tuple(init_state)

    
#creating lists of operators and corresponding positions
  def split(self):
    pauli_lst = []
    pos_lst = []
    prev_int = False
    for k in self.string:
        if k.isdigit():
            if not prev_int:
                pos_lst.append(k)
            else:
                pos_lst[-1] += k
            prev_int = True
        else:
            pauli_lst.append(k)
            prev_int = False
    return pos_lst, pauli_lst

  def to_parray(self):
    parray = np.zeros((self.N,2,2),dtype = complex)
    for j in range(self.N):
      parray[j,:,:] = Pauli[self.string[j]]

    return pauli_array(parray,self.N,self.factor)
          

  
  def matrix_repr(self):
    Kron = 1
    for j in range(self.N):
        Kron = np.kron(Kron, Pauli[self.string[j]])
    return Kron*self.factor

 
class pauli_array:
    def __init__(self,parray,N, factor = 1):
        self.N = N
        self.parray = parray
        self.factor = factor
        self.init_state = np.array([[1,0]]*N)

    def __str__(self):
        return ".   factor: "+str(self.factor)

    #define multiplying by a constant (on left hand side)
    def __rmul__(self, c):
        return pauli_array(self.parray,self.N, c*self.factor)

    def __pow__(self,c):
        if c == 0:
            id = np.broadcast_to(np.eye(2), (self.N,2,2))
            return pauli_array(id,self.N)
        elif c ==1:
            return self   
     
    def __mul__(self,x):
        return pauli_array(np.matmul(self.parray, x.parray),self.N,self.factor*x.factor)

    def matrix_repr(self):
        return reduce(np.kron, list(self.parray))
    
    def state(self,init_state=0):
        state = np.einsum('nij,ni->nj', self.parray, self.init_state)
        factor = np.prod(np.sum(state,axis = 1))
        return factor, tuple(np.where(state[:,0]==1,0,1))


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





def place_ones(size, order):
    for i in range(order+1):
        for positions in combinations(range(size), i):
            p = [0] * size

            for i in positions:
                p[i] = 1
            yield p


@timing
def s_dict(N, ansatz, K, order):
  start = time()
  s_dict = {} #keys: possible bitstrings, values dictionary with orders
  T_K = pull_cliffords_through(ansatz, K, N)
  L = len(ansatz)
#   print(T_K,"T_K", time()-start)
  for i in place_ones(len(ansatz), order): #loop through all

    #calculate state that is produced by T_i
    pauli_string = power_product(T_K[::-1], i[::-1])
    factor, state = pauli_string.state()
    
    #calculate magnitude of term
    term = (1j)**sum(i)*factor
    
    #check whether binary string is in dictionary, otherwise add
    if state not in s_dict:
      s_dict[state] = ([list(i)],[term])
    else:
      current = s_dict[state]
      current[0].append(list(i))
      current[1].append(term)
  time_dict = time()-start
  # print("build dict",time_dict)
    
  #make np.array
  for st in s_dict:
    lst = s_dict[st]
    s_dict[st] = (np.array(lst[0]),np.array(lst[1]))
  # print("make array", time()-time_dict-start)
  return s_dict


@timing
def s_dict2(N, ansatz, K, order):
    T_K = pull_cliffords_through(ansatz, K, N)
    firstorder = [0]*len(ansatz)
    s_dict = {tuple([0]*N) : ([firstorder], [1])}
    #index of last nonzeror element in firstorder
    def next_order(list, last_nonzero,add_order, calc = True, factor = 1, state = [0]*N):
        if calc:
            
            pauli_operator = T_K[last_nonzero]
            f, state = pauli_operator.state(np.array(state))
            factor = 1j*factor*f

            #add to dictionary
            if state not in s_dict:
                s_dict[state] = ([list],[factor])
            else:
                current = s_dict[state]
                current[0].append(list)
                current[1].append(factor)

        if add_order > 0:
            if last_nonzero < len(list)-1:
                l = list.copy()
                l[last_nonzero+1] = 1
                next_order(l, last_nonzero+1, add_order-1, factor = factor, state = state)

            if last_nonzero < len(list)-1:
                next_order(list, 1+last_nonzero, add_order, calc = False, factor = factor, state = state)

    next_order(firstorder, -1, order, calc = False)
    for st in s_dict:
        lst = s_dict[st]
        s_dict[st] = (np.array(lst[0]),np.array(lst[1]))
    return s_dict


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


def energy(thetas,ansatz, s_dict,G_K, order):
  E = 0
  s_dict1 = s_dict
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

def overlap(theta_t, theta_a, K_init, K_a, ansatz):
    wave_1 = psi(theta_t, ansatz, K_init)
    wave_2 = psi(theta_a, ansatz, K_a)
    
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

def QAOA(N:int, L:int, array_method = False):
    ansatz = []
    assert L%2 == 0, "L is odd"

    XX_layer = [pauli(f"X{i}X{i+1}",N) for i in range(N-1)] # not yet flattened circuit
    Z_layer = [pauli(f"Z{i}", N) for i in range(N)]

    for i in range(int(L/2)):
        ansatz += XX_layer
        ansatz += Z_layer

    if array_method:
        ansatz = [a.to_parray() for a in ansatz]
    return ansatz

def TUCC(N:int, Layers:int,array_method=False):
    XY_layer = [pauli(f"X{i}Y{i+1}", N) for i in range(N-1)]
    ansatz = XY_layer*Layers
    if array_method:
        ansatz = [a.to_parray() for a in ansatz]
    return ansatz



#TFIM

def TFIM(N, X_h, Z_h=-1,array_method=False):
    Z_terms = [pauli(f'Z{i}',N,Z_h) for i in range(N)]
    X_terms = [pauli(f'X{i}X{i+1}',N,X_h) for i in range(N-1)]
    H = X_terms + Z_terms

    if array_method:
        H = [h.to_parray() for h in H]
    return H



def find_new_branch(K_tree, node, K_path, shuffle = False):

    node_up = node[:-1]
    if node == (0,):
        return "whole tree has been searched"
    N_magic = K_tree[node]["N_magic"]
    
    updated = False

    #randomly shuffle magic gates
    magic_gates = list(range(N_magic))
    if shuffle:
        random.shuffle(magic_gates)

    #loop through randomly ordered magic-gates
    for i in magic_gates:
        tree_i = node+(i,)
        K_i = K_tree[tree_i]["K"]
        if not K_tree[tree_i]['seen'] and K_i not in K_path:
            updated = True
            return tree_i

    if not updated:
        return find_new_branch(K_tree, node_up, K_path)

def output(K_tree, optim_node, termination, matrix_min, ansatz, N, H, log = True):
    #get node with lowest energy
    K_best = K_tree[optim_node]
    angles = K_best['angles']
    K = K_best['K']
    E_a = Energy_matrix(angles, N, H, ansatz, K)

    #print information about best K-cell
    if log:
        print("\n")
        print("-"*40)
        print("\n RESULT \n")
        print(f"{'Termination cause' + ':':<25}{f'{termination}'}\n")
        for key, value in K_best.items():
            print(f"{f'{key}:':<25}{f'{value}'}\n")

        #E_t equals true energy, with angles found by minimizer 
        #E_a equals approximated energy, with angles found by approximation

        print(f"{'E_t:':<25} {f'{matrix_min.fun}'}\n")
        
        print(f"{'E_a:':<25} {f'{E_a}'}\n")
        
    return K_best, E_a
    


def find_K(N, ansatz, H, iterations, order, boundary = "hypersphere",log=True, matrix_min = None):

    H_m = sum([h.matrix_repr() for h in H])
    ansatz_m = [a.matrix_repr() for a in ansatz]
    N_K = iterations #number of iterations
    K_init = list(np.random.randint(4, size = len(ansatz))) #random initial K-cell
    K = K_init.copy()
    theta_init = [1/np.pi/16]*len(ansatz) #intial angles
    K_tree = {(0,):{"K":K,"angles":theta_init},(0,0):{"K":K, "seen":False}}
    Energy_prev = np.inf
    termination = "Maximum number of iters has been reached"
    curr_node = (0,0)
    optim_node = (0,0) #node with lowest energy


    if boundary == "hypercube":#set boundary to hypercube
        bounds = [(-np.pi/8,np.pi/8)]*len(ansatz)

    elif boundary == "hypersphere":#set boundary to hypersphere

        #define constraint of hypersphere
        def constraint(thetas):
            corner = np.sqrt(len(thetas)*(np.pi/8)**2)
            norm = np.linalg.norm(thetas)
            return corner - norm

        #define constraint to pass into scipy.minimize
        con = {'type':"ineq", 'fun':constraint}
        

    epsilon = 1e-3 # angle close enough to magic border of pi/8
    E_epsilon = 1e-5 # Energy must not increase with more than this value
    K_path = [] #store all K's visited

    #calculate global minimum
    start = time()
    if matrix_min is None:
        matrix_min = scipy.optimize.minimize(Energy_matrix, theta_init, jac = False, args = (N,H_m,ansatz_m, K))
    end = time()
    print(f"{'Time to find local minimum:':<25} {f'{end-start}'}\n")

    if log: print("LOG")

    for iter in range(N_K):

        K = K_tree[curr_node]["K"]

        s = s_dict2(N, ansatz, K, order)
        G_K = G_k(N, H, ansatz,K)



        #previous minimized set of angles
        node_above= K_tree[curr_node[:-1]]
        prev_angles_global = local_to_global_angle(node_above["angles"], node_above["K"])

        #translate to angles in current K_cell
        init_angles = global_to_local_angle(prev_angles_global, K)

        #calculate energy
        if boundary == "hypercube": #boundary set to cube
            result = scipy.optimize.minimize(energy, init_angles,jac = False, args = (ansatz,s,G_K,order),bounds=bounds)

        elif boundary == "hypersphere": #boundary set to sphere
            result = scipy.optimize.minimize(energy, init_angles,jac = False, args = (ansatz,s,G_K,order),constraints=con)
        
        else:
            raise ValueError("Invalid boundary condition was given. hypercube or hypersphere")

        #get indices of magic gates
        magic_indices = np.where(np.pi/8 -  np.abs(result.x)< epsilon)

        N_magic = len(magic_indices[0])
        Energy = result.fun


        #add data to dictionary
        K_tree[curr_node]["seen"] = True
        K_tree[curr_node]["N_magic"]= N_magic
        K_tree[curr_node]["energy"] = Energy
        K_tree[curr_node]["angles"] = result.x

        #add K-vector corresponding to curr_node to K_path
        K_path += [K.copy()]

        if log:
            print("-"*30)
            print(f"iteration: {iter}")
            for key, value in K_tree[curr_node].items():
                print(f"{f'{key}:':<25}{f'{value}'}")

        #if Energy increases, find new branch skip iteration
        if Energy - Energy_prev > E_epsilon:
            new_node = find_new_branch(K_tree, curr_node[:-1], K_path)
            if type(new_node)!=tuple:
                termination = "Whole tree has been explored"
                break
            else: 
                curr_node = new_node
                continue


        #include new branches
        for i in range(N_magic):
            magic_index = magic_indices[0][i]
            sign = np.sign(result.x[magic_index])
            K_i = K_tree[curr_node]["K"].copy()
            K_i[magic_index] = K_i[magic_index]+int(sign)*1
            K_tree[curr_node+(i,)]={"K":K_i, "seen":False}
        
        #choose new branch, only fires when Energy decreases or slightly increases
        assert Energy - Energy_prev < E_epsilon, "Energy > Energy_prev??"

        #update best node, if energy decreased
        if Energy - Energy_prev < 0:
            optim_node = curr_node

        #update current node
        new_node = find_new_branch(K_tree, curr_node, K_path)
        if type(new_node)!=tuple:
            termination = "Whole tree has been explored"
            break
        else:
            curr_node = new_node

        #update previous energy
        Energy_prev = Energy

    #ouput
    out, E_a = output(K_tree, optim_node, termination, matrix_min,ansatz_m,N,H_m, log)

    #Ratio between number of nfev theta_init to theta_t and theta_a to theta_t
    theta_a = out['angles']
    K_a = out["K"]
    appr_min = scipy.optimize.minimize(Energy_matrix, theta_a, jac = False, args = (N,H_m,ansatz_m, K_a))


    nfev_ratio = matrix_min.nfev/appr_min.nfev
    E_a_t = appr_min.fun #
    E_t = matrix_min.fun
    Overlap = overlap(matrix_min.x, theta_a, K_init, K_a, ansatz_m)

    if log:
         print(f"{'Overlap <Ψ_t|Ψ_a>:':<25} {f'{Overlap}'}\n")
    print(f"nfev_t/nfev_a: {matrix_min.nfev}/{appr_min.nfev} = {matrix_min.nfev/appr_min.nfev}")
    
    return nfev_ratio, Overlap, E_a, E_t, E_a_t, matrix_min

  

class optimizer:
    def __init__(self, func, x0, args = (), boundary = "hypersphere", epsilon= 1e-3):
        self.func = func
        self.x0 = np.asarray(x0)
        self.boundary = boundary
        self.epsilon = epsilon
        self.args = args

    def callback(self, x):
        if self.boundary == "hypersphere":
            corner = np.sqrt(len(x)*(np.pi/8)**2)
            norm = np.linalg.norm(x)
            if corner - norm < self.epsilon:
                #quit optimization
                print("Optimization stopped because of boundary condition")
                return True
    def optimize(self):
        opt = scipy.optimize.minimize(self.func, self.x0, jac = False, args = self.args)
        return opt



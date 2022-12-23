from Func import *


def place_ones(size, order):
    for i in range(order+1):
        for positions in combinations(range(size), i):
            p = [0] * size

            for i in positions:
                p[i] = 1
            yield p


#old slow version
@timing
def s_dict_old(N, ansatz, K, order):
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


    # if boundary == "hypercube":#set boundary to hypercube
    #     bounds = [(-np.pi/8,np.pi/8)]*len(ansatz)

    # elif boundary == "hypersphere":#set boundary to hypersphere

    #     #define constraint of hypersphere
    #     def constraint(thetas):
    #         corner = np.sqrt(len(thetas)*(np.pi/8)**2)
    #         norm = np.linalg.norm(thetas)
    #         return corner - norm

    #     #define constraint to pass into scipy.minimize
    #     con = {'type':"ineq", 'fun':constraint}

from Func import *
from deprecated import *
import cirq
Pauli = {"I":np.array([[1,0],[0,1]]), "X":np.array([[0,1],[1,0]]), "Y":np.array([[0,-1j],[1j,0]]), "Z":np.array([[1,0],[0,-1]])}

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

  def cirq_repr(self, qubits = None):
    if qubits == None:
        qubits = cirq.LineQubit.range(self.N)
    return cirq.PauliString(dict(zip(qubits, self.string)), coefficient = self.factor)


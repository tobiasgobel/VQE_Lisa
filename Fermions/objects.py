import numpy as np

class state:
    def __init__(self, n, x, factor = 1):
        self.n = n
        self.x = x
        self.factor = factor
    def __str__(self):
        repr = "state(" + str(self.n) + ", " + str(self.x) + ", " + str(self.factor) + ")"
        return repr
    def __repr__(self):
        repr = "state(" + str(self.n) + ", " + str(self.x) + ", " + str(self.factor) + ")"
        return repr
    def __mul__(self, other):
        if type(other) == int or type(other) == float or type(other) == complex:
            return state(self.n, self.x, self.factor*other)
        elif type(other) == state:
            return state(self.n, self.x + other.x, self.factor*other.factor)
    def __rmul__(self, other):
        if type(other) == int or type(other) == float or type(other) == complex:
            return state(self.n, self.x, self.factor*other)
        elif type(other) == state:
            return state(self.n, self.x + other.x, self.factor*other.factor)

    def xflip(self, i):
        x = self.x.copy()
        i = int(i)
        x[i] = 1 - x[i]
        return state(self.n, x, self.factor)

    def yflip(self, i):
        x = self.x.copy()
        i = int(i)
        x[i] = 1 - x[i]
        sign = (-1)**(x[i])
        return state(self.n, x, sign*1j*self.factor)

class monomial:
    def __init__(self, n: int, positions: list, factor: np.complex128 = 1, index = None):
        self.positions = positions
        self.factor = factor
        self.n = n
        self.index = index

    def __str__(self):
        repr = "monomial("+ str(self.factor) + ", " +  str(self.n) + ", "
        for i in range(len(self.positions)):
            repr += "C"+ str(self.positions[i])
        repr += ")"
        return repr


    def __repr__(self):
        repr = "monomial("+ str(self.factor) + ", " + str(self.n) + ", "
        for i in range(len(self.positions)):
            repr += "C"+ str(self.positions[i])
        repr += ")"
        return repr
    def __len__(self):
        return len(self.positions)

    #returns 0 if parity is even, 1 if parity is odd
    @property
    def parity(self):
        return len(self.positions)%2

    def parallel_matrix(self):
        mat = np.zeros((2*self.n, len(self.positions)))
        for i,p in enumerate(self.positions):
            mat[p, i] = 1
        return mat

class circuit:
    def __init__(self, monomials: list):
        self.monomials = monomials
        self.GaussianIndices, self.NonGaussianindices = self.indices()
        self.gates = self.rotations()

    def __str__(self):
        repr = "circuit("
        for i in range(len(self.monomials)):
            repr += str(self.monomials[i]) + ", "
        repr += ")"
        return repr
    
    def __repr__(self):
        repr = "circuit("
        for i in range(len(self.monomials)):
            repr += str(self.monomials[i]) + ", "
        repr += ")"
        return repr

    def __add__(self, other):
        return circuit(self.monomials + other.monomials)
    def __len__(self):
        return len(self.monomials)
    def __getitem__(self,index):
        return self.monomials[index]

    def indices(self):
        gaussian=[]
        nongaussian=[]
        for monomial in self.monomials:
            if len(monomial.positions)==2:
                gaussian.append(monomial)
            elif len(monomial.positions)==4:
                nongaussian.append(monomial)
            else:
                raise ValueError("invalid"+str(monomial))
        return gaussian, nongaussian

    def rotations(self):
        l = [GaussianRotation(m) if len(m.positions)==2 else NonGaussianUnitary(m) for m in self.monomials]
        return l

    def evaluate(self, angles: list, state):
        return state(self, angles, state)

# Class of Gaussian unitaries with a method to multiply
class GaussianUnitary:
    def __init__(self, matrix_repr=None):
        self.matrix_repr = matrix_repr

    def __mul__(self, other):
        if type(other)==int:
            return GaussianUnitary(self.matrix_repr*other)
        else:
            matrix = np.dot(self.matrix_repr, other.matrix_repr)
            return GaussianUnitary(matrix)

    def act_on_monomial(self, monomial, dagger = False):
        m = monomial.parallel_matrix()
        if dagger:
            return self.matrix_repr.T @ m
        else:
            return self.matrix_repr @ m

# Child class of Gaussian unitary
class GaussianFlip(GaussianUnitary):
    def __init__(self, monomial):
        super().__init__()
        self.monomial = monomial
        self.N = monomial.n
        self.matrix_repr = self.compute_matrix_repr()

    def compute_matrix_repr(self):
        matrix = np.eye(2 * self.N)
        for position in monomial.positions:
            matrix*=-1
            matrix[position, position] *= -1
        return matrix

# Child class of Gaussian unitary
class GaussianRotation(GaussianUnitary):
    def __init__(self, monomial, angle = None):
        super().__init__()
        self.monomial = monomial.positions
        self.N = monomial.n
        self.matrix_repr =  self.compute_matrix_repr(angle)
    
    def compute_matrix_repr(self, angle):
        if angle is not None:
            matrix = np.eye(2 * self.N)
            assert len(self.monomial) == 2, "Invalid monomial for GaussianRotation"
            pos1, pos2 = self.monomial

            matrix[pos1, pos1] = matrix[pos2, pos2] = np.cos(angle*2)
            matrix[pos2, pos1] = -np.sin(angle*2)
            matrix[pos1, pos2] = np.sin(angle*2)
            self.matrix_repr = matrix
            return matrix
        else:
            return None


#Non-Gaussian gate
class NonGaussianUnitary:
    def __init__(self, monomial):
        self.monomial = monomial
        self.N = monomial.n


n=8
l = [monomial(n, [1,2]), monomial(n,[3,4]), monomial(n,[6,7,8,9])]
print(circuit(l).gates)

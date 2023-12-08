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
    def __init__(self, n: int, positions: list, factor: np.complex128 = 1):
        self.positions = positions
        self.factor = factor
        self.n = n

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
    

    #returns 0 if parity is even, 1 if parity is odd
    @property
    def parity(self):
        return len(self.positions)%2


class circuit:
    def __init__(self, monomials: list):
        self.monomials = monomials

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

    def evaluate(self, angles: list, state):
        return state(self, angles, state)

# Class of Gaussian unitaries with a method to multiply
class GaussianUnitary:
    def __init__(self, matrix_repr=None):
        self.matrix_repr = matrix_repr

    def __mul__(self, other):
        matrix = np.dot(self.matrix_repr, other.matrix_repr)
        return GaussianUnitary(matrix)

# Child class of Gaussian unitary
class GaussianFlip(GaussianUnitary):
    def __init__(self, monomial):
        super().__init__()
        assert len(monomial.positions) == 1
        self.monomial = monomial.positions
        self.N = monomial.n
        self.matrix_repr = self.compute_matrix_repr()

    def compute_matrix_repr(self):
        matrix = np.eye(2 * self.N)
        matrix[self.monomial, self.monomial] = -1
        return matrix

# Child class of Gaussian unitary
class GaussianRotation(GaussianUnitary):
    def __init__(self, monomial, angle):
        super().__init__()
        self.angle = angle
        self.monomial = monomial.positions
        self.N = monomial.n
        self.matrix_repr = self.compute_matrix_repr()


    def compute_matrix_repr(self):
        matrix = np.eye(2 * self.N)
        assert len(self.monomial) == 2, "Invalid monomial for GaussianRotation"
        pos1, pos2 = self.monomial
        matrix[pos1, pos1] = matrix[pos2, pos2] = np.cos(self.angle)
        matrix[pos2, pos1] = -np.sin(self.angle)
        matrix[pos1, pos2] = np.sin(self.angle)
        return matrix

# Example usage
a = GaussianFlip(monomial(4, [2]))
b = GaussianRotation(monomial(4, [1, 2]), 0.2)

import numpy as np



class monomial:
    def __init__(self, n: int, positions: list, factor: np.complex128):
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
    
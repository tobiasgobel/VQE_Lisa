from objects import *

def Pfaffian(matrix):
    """Returns the Pfaffian of a matrix."""
    if len(matrix) % 2 != 0:
        raise ValueError("Matrix must have even dimension.")
    if matrix != matrix.T:
        raise ValueError("Matrix must be symmetric.")
    if len(matrix) == 2:
        return matrix[0,1]
    else:
        return sum([(-1)**(i+1) * matrix[0,i] * Pfaffian(matrix[1:,1:i]) for i in range(1,len(matrix))])


def state(circuit, n,  )
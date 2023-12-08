from objects import *

def Pfaffian(matrix):
    if matrix.shape == (2,2):
        return matrix[0,1]
    else:
        return np.linalg.det(matrix)**2

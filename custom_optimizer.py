import scipy
import numpy as np

class E_optimizer:
    def __init__(self, func, x0, args = (), boundary = "hypersphere", epsilon= 1e-4):
        self.func = func
        self.x0 = np.asarray(x0)
        self.boundary = boundary
        self.epsilon = epsilon
        self.args = args

    def callback(self, x, *args):
        if self.boundary == "hypersphere":
            corner = np.sqrt(len(x)*(np.pi/8)**2)
            norm = np.linalg.norm(x)
            if corner - norm < self.epsilon:
                #quit optimization
                print("Optimization stopped because of boundary condition")
                return True
    
    def optim(self):
        opt = scipy.optimize.minimize(self.func, self.x0, jac = False, args = self.args, method = "trust-constr")
        return opt
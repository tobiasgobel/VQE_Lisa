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
                return True
    
    def constraint(self, thetas):
        corner = np.sqrt(len(thetas)*(np.pi/8)**2)
        norm = np.linalg.norm(thetas)
        return corner - norm

    def optim(self):
        con = {'type':"ineq", 'fun':self.constraint}
        opt = scipy.optimize.minimize(self.func, self.x0, jac = False,args = self.args, constraints = con)
        return opt
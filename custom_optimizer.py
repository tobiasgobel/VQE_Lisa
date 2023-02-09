import scipy
import numpy as np
import matplotlib.pyplot as plt

#Optimization class for optizing inside the K-cell
class E_optimizer:
    def __init__(self, func, x0, args = (), boundary = "hypersphere", epsilon= 1e-4, method = "SLSQP"):
        self.func = func
        self.x0 = np.asarray(x0)
        self.boundary = boundary
        self.epsilon = epsilon
        self.args = args
        self.method = method
        self.plot = False

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
        opt = scipy.optimize.minimize(self.func, self.x0, jac = False,args = self.args, constraints = con, method = self.method, options={'rhobeg':0.001, 'disp':True})
        return opt


#Optimization class for optimizing between theta_a and theta_t
class E_optim_cirq:
    def __init__(self, func, x0, args = (), method = "COBYLA", plot=False):
        self.func = func
        self.x0 = np.asarray(x0)
        self.args = args
        self.method = method
        self.plot = plot
        self.angles = []
        self.energy = []
        
    def callback(self, x):
        self.angles.append(x)
        self.energy.append(self.func(x, *self.args))

    def optim(self):
        opt = scipy.optimize.minimize(self.func, self.x0,args = self.args, method = self.method, callback = self.callback)
        if self.plot:
            iterations = len(self.angles)
            print(self.angles)
            self.angles = np.array(self.angles).reshape(iterations, len(self.x0))
            plt.ylabel("Angle")
            plt.xlabel("Iteration")
            plt.plot(self.angles)
            plt.savefig("angles.png")
            plt.clf()

            plt.ylabel("Log(Energy)")
            plt.xlabel("Iteration")
            plt.plot(np.log(self.energy))
            plt.savefig("Energy.png")
            plt.clf()

        return opt



    


from helper_functions import *
from collections import defaultdict
from itertools import combinations
from hamiltonians import *

from itertools import chain, combinations

def place_ones(size, order):
    for i in range(order+1):
        for positions in combinations(range(size), i):
            p = list(positions)
            yield p


def energy_non_gaussian(n, H, circ, angles, order):
    E = 0
    #get indices of Gaussian/non-Gaussian gates
    gauss_indices = circ.GaussianIndices
    non_gauss_indices = circ.NonGaussianindices

    #For every term in the Hamiltonian
    for h in H:
        #lightcone for observable h
        lc_h, lc_h_indices = lightcone(h, circ)

        #all gaussian monomials in the lightcone
        lc_h_gaussian = list(set(lc_h).intersection(set(gauss_indices)))

        #all non-gaussian monomials in the lightcone
        lc_h_nongaussian = list(set(lc_h).intersection(set(non_gauss_indices)))


        #construct c-tilde
        U_t_g = 1
        for t in reversed(lc_h_gaussian):
            angle = angles[t.index]
            U_t_g = GaussianRotation(t, angle = angle)* U_t_g
        C_tilde_J = U_t_g.act_on_monomial(h, dagger= True)

        #gaussian lightcones of non-gaussian gates
        n_gaussian_gates = len(lc_h_nongaussian)
        lc_t_g = np.zeros((n_gaussian_gates, 2*n, 4))#assuming all non-g gates are of length 4
        for j, nongauss in enumerate(lc_h_nongaussian):
            i = nongauss.index
            lc_i = lightcone(nongauss, circ[i:], gaussian=True)

            #Calculate C(t) tildas
            #Room for improvement, as rotations can be reused
            U_t_g = 1
            for t in reversed(lc_i[0]):
                angle = angles[t.index]
                U_t_g = GaussianRotation(t, angle = angle)* U_t_g
            lc_t_g[j] = U_t_g.act_on_monomial(nongauss, dagger = True)

        #sum over all pairs of subsets
        for i in place_ones(n_gaussian_gates, order):
            for j in place_ones(n_gaussian_gates, order):
                #take appropriate indices 
                forward = np.flipud(lc_t_g[i,:,:])
                backward = lc_t_g[j,:,:]

                #collapse forward and backward objects
                forward = np.einsum('ijk->jik',forward).reshape(2*n, -1)
                backward = np.einsum('ijk->jik',backward).reshape(2*n, -1)

                #concatenate
                rotated_Cs = np.concatenate((forward,C_tilde_J,backward), axis =1)
                print(rotated_Cs)
                


                




n = 5
H = fermionic_TFIM(n)
circ = circuit(ansatz_TFIM(n)+[monomial(n, [0,1,2,3])]+[monomial(n, [0,1,2,4])])
print(circ)
angles = np.linspace(0.5, 2*np.pi, len(circ))
order = 4
energy_non_gaussian(n, H, circ, angles, order)
        


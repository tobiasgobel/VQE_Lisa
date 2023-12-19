from helper_functions import *
from collections import defaultdict
from itertools import combinations


from itertools import chain, combinations

def powerset(lst, k):
    return chain.from_iterable(combinations(lst, r) for r in range(k + 1))


def energy_non_gaussian(H, circ, angles, order):
    E = 0
    #get indices of Gaussian/non-Gaussian gates
    gauss_indices = circ.GaussianIndices
    non_gauss_indices = circ.NonGaussianindices

    #Build dictionary
    for h in H:
        #lightcone for observable h
        lc_h = lightcone(h, circ)

        #all non-gaussian monomials in the lightcone
        lc_h_gaussian = list(set(lc_h).intersection(set(gauss_indices)))

        #all non-gaussian monomials in the lightcone
        lc_h_nongaussian = list(set(lc_h).intersection(set(non_gauss_indices)))

        #create all subsets of non-gaussian monomials in the lightcone
        all_subsets = list(powerset(lc_h_nongaussian, order))

        #construct list of c-tildas
        C_tildas = []
        for t in lc_h_nongaussian:
            t_lightcone = lightcone(circuit[t], circuit[t:], gaussian = True)
            raise NotImplementedError

        #sum over all 
        for i in all_subsets:
            for j in all_subsets:

        
        


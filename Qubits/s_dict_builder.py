from Func import *
from Energy_funcions import *
#@timing
def s_dict(N, ansatz, K, order, prepare_x_basis = False, cliffords_pulled_through = False):
    
    
    T_K = pull_cliffords_through(ansatz, K, N)

    if prepare_x_basis:
        T_K = T_K_prepare_x_basis(N, T_K)
    
    assert len(ansatz) == len(K)

    firstorder = [0]*len(ansatz)
    s_dict = {tuple([0]*N) : ([firstorder], [1])}

    
    #index of last nonzeror element in firstorder
    def next_order(list, last_nonzero,add_order, calc = True, factor = 1, state = [0]*N):
        if calc:
            pauli_operator = T_K[last_nonzero]
            f, state = pauli_operator.state(np.array(state))
            factor = 1j*factor*f

            #add to dictionary
            if state not in s_dict:
                s_dict[state] = ([list],[factor])
            else:
                current = s_dict[state]
                current[0].append(list)
                current[1].append(factor)

        if add_order > 0:
            if last_nonzero < len(list)-1:
                l = list.copy()
                l[last_nonzero+1] = 1
                next_order(l, last_nonzero+1, add_order-1, factor = factor, state = state)
                next_order(list, 1+last_nonzero, add_order, calc = False, factor = factor, state = state)

    next_order(firstorder, -1, order, calc = False)

    for st in s_dict:
        lst = s_dict[st]
        s_dict[st] = (np.array(lst[0]),np.array(lst[1], dtype = np.complex128))

    return s_dict



#function for calculating the product of activated tangents of a certain term
def tangent_product(activation, tan_thetas):
    activated_thetas = np.where(np.array(activation) == 1, tan_thetas, 1)
    product = np.prod(activated_thetas)
    return product


#function that builds the s-dict based on the size of different angles
def s_dict_tree(N, ansatz, K, thetas, treshold, prepare_x_basis = False):
    T_K = pull_cliffords_through(ansatz, K, N)
    if prepare_x_basis:
        T_K = T_K_prepare_x_basis(N, T_K)
    
    firstorder = [0]*len(ansatz)
    s_dict = {tuple([0]*N) : ([firstorder], [1])}
    tan_thetas = np.tan(thetas)

    
    #index of last nonzeror element in firstorder
    def next_order(list, last_nonzero, calc = True, factor = 1, state = [0]*N, prod = 1):
        if calc:
            pauli_operator = T_K[last_nonzero]
            f, state = pauli_operator.state(np.array(state))
            factor = 1j*factor*f

            #add to dictionary
            if state not in s_dict:
                s_dict[state] = ([list],[factor])
            else:
                current = s_dict[state]
                current[0].append(list)
                current[1].append(factor)

        #prod = tangent_product(list, tan_thetas)
        if abs(prod) > treshold:
            if last_nonzero < len(list)-1:
                l = list.copy()
                l[last_nonzero+1] = 1
                next_order(l, last_nonzero+1, factor = factor, state = state, prod = prod*tan_thetas[last_nonzero+1])
                next_order(list, 1+last_nonzero, calc = False, factor = factor, state = state, prod = prod)

    next_order(firstorder, -1, calc = False)

    for st in s_dict:
        lst = s_dict[st]
        s_dict[st] = (np.array(lst[0]),np.array(lst[1]))

    return s_dict


def s_dicts_lightcones(N, ansatz, H, K, order):
    sdicts = []
    lc = []
    T_K = pull_cliffords_through(ansatz, K, N)
    for h in H:
        lc_indices = lightcone([h], T_K, order)
        lc_gates = [ansatz[i] for i in lc_indices]
        K_indexed = [K[i] for i in lc_indices]
        sdicts.append(s_dict(N, lc_gates, K_indexed, order))
        lc.append(lc_indices)
    return sdicts, lc


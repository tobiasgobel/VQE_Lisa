from Func import *
from Energy_funcions import *
#@timing
def s_dict(N, ansatz, K, order, prepare_x_basis = False):
    T_K = pull_cliffords_through(ansatz, K, N)
    if prepare_x_basis:
        T_K = T_K_prepare_x_basis(N, T_K)
    
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

            if last_nonzero < len(list)-1:
                next_order(list, 1+last_nonzero, add_order, calc = False, factor = factor, state = state)

    next_order(firstorder, -1, order, calc = False)

    for st in s_dict:
        lst = s_dict[st]
        s_dict[st] = (np.array(lst[0]),np.array(lst[1]))

    return s_dict

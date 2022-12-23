from Func import *
from deprecated import *
from s_dict_builder import *
from custom_optimizer import *
from pauli_objects import *
from Energy_funcions import *
from Ansatzes_Hamiltonians import *

def find_new_branch(K_tree, node, K_path, shuffle = False):

    node_up = node[:-1]
    if node == (0,):
        return "whole tree has been searched"
    N_magic = K_tree[node]["N_magic"]
    
    updated = False

    #randomly shuffle magic gates
    magic_gates = list(range(N_magic))
    if shuffle:
        random.shuffle(magic_gates)

    #loop through randomly ordered magic-gates
    for i in magic_gates:
        tree_i = node+(i,)
        K_i = K_tree[tree_i]["K"]
        if not K_tree[tree_i]['seen'] and list(K_i) not in K_path:
            updated = True
            return tree_i

    if not updated:
        return find_new_branch(K_tree, node_up, K_path)

def output(K_tree, optim_node, termination, matrix_min, ansatz, N, H, log = True, HVA = False):
    #get node with lowest energy
    K_best = K_tree[optim_node]
    angles = K_best['angles']
    K = K_best['K']
    E_a = Energy_matrix(angles, N, H, ansatz, K, HVA = HVA)

    #print information about best K-cell
    if log:
        print("\n")
        print("-"*40)
        print("\n RESULT \n")
        print(f"{'Termination cause' + ':':<25}{f'{termination}'}\n")
        for key, value in K_best.items():
            print(f"{f'{key}:':<25}{f'{value}'}\n")

        #E_t equals true energy, with angles found by minimizer 
        #E_a equals approximated energy, with angles found by approximation

        print(f"{'E_t:':<25} {f'{matrix_min.fun}'}\n")
        
        print(f"{'E_a:':<25} {f'{E_a}'}\n")
        
    return K_best, E_a
    

@timing
def find_K(N, ansatz, H, iterations, order, boundary = "hypersphere",log=True, matrix_min = None, HVA = False):

    H_m = sum([h.matrix_repr() for h in H])
    ansatz_m = [a.matrix_repr() for a in ansatz]
    N_K = iterations #number of iterations

     #random initial K-cell
    if HVA:
        theta_init = [1/np.pi/16]*HVA
        K_init = list(np.random.randint(0,4, HVA))
        K_init = distribute_over_gates(HVA, N, K_init)
    else:
        theta_init = [1/np.pi/16]*len(ansatz)
        K_init= list(np.random.randint(4, size = len(ansatz)))

    
    K = K_init.copy()

    K_tree = {(0,):{"K":K,"angles":theta_init},(0,0):{"K":K, "seen":False}}
    Energy_prev = np.inf
    termination = "Maximum number of iters has been reached"
    curr_node = (0,0)
    optim_node = (0,0) #node with lowest energy
    epsilon = 1e-3 # angle close enough to magic border of pi/8
    E_epsilon = 1e-5 # Energy must not increase with more than this value
    K_path = [] #store all K's visited

    #calculate minimum
    start = time()
    if matrix_min is None:
        matrix_min = scipy.optimize.minimize(Energy_matrix, theta_init, jac = False, args = (N,H_m,ansatz_m, K, HVA))
    end = time()
    print(f"{'Time to find local minimum:':<25} {f'{end-start}'}\n")
    if log: print("LOG")

    for iter in range(N_K):

        K = K_tree[curr_node]["K"]
        s = s_dict(N, ansatz, K, order)
        G_K = G_k(N, H, ansatz,K)

        #previous minimized set of angles
        node_above= K_tree[curr_node[:-1]]


        #calculate energy
        if HVA:
            args = (s, G_K, order, HVA)
            prev_angles_global = local_to_global_angle(node_above["angles"], shorten_vector(HVA, N, node_above["K"]))
            init_angles = global_to_local_angle(prev_angles_global, shorten_vector(HVA, N, K))
        else:   
            args = (s, G_K, order)
            prev_angles_global = local_to_global_angle(node_above["angles"], node_above["K"])
            #translate to angles in current K_cell
            init_angles = global_to_local_angle(prev_angles_global, K)

        print(K)
        optimizer = E_optimizer(energy, init_angles, args = args)
        result = optimizer.optim()

        #get indices of magic gates
        magic_indices = np.where(np.pi/8 -  np.abs(result.x)< epsilon)

        N_magic = len(magic_indices[0])
        Energy = result.fun


        #add data to dictionary
        K_tree[curr_node]["seen"] = True
        K_tree[curr_node]["N_magic"]= N_magic
        K_tree[curr_node]["energy"] = Energy
        K_tree[curr_node]["angles"] = result.x

        #add K-vector corresponding to curr_node to K_path
        K_path += [list(K).copy()]

        if log:
            print("-"*30)
            print(f"iteration: {iter}")
            for key, value in K_tree[curr_node].items():
                print(f"{f'{key}:':<25}{f'{value}'}")

        #if Energy increases, find new branch skip iteration
        if Energy - Energy_prev > E_epsilon:
            new_node = find_new_branch(K_tree, curr_node[:-1], K_path)
            if type(new_node)!=tuple:
                termination = "Whole tree has been explored"
                break
            else: 
                curr_node = new_node
                continue
        

        #include new branches
        for i in range(N_magic):
            magic_index = magic_indices[0][i]
            sign = np.sign(result.x[magic_index])
            K_i = K_tree[curr_node]["K"].copy()
            if HVA:
                K_i = shorten_vector(HVA, N, K_i)
                K_i[magic_index] = K_i[magic_index]+int(sign)*1
                K_i = distribute_over_gates(HVA, N, K_i)
            else:
                K_i[magic_index] = K_i[magic_index]+int(sign)*1

            K_tree[curr_node+(i,)]={"K":K_i, "seen":False}
        
        #choose new branch, only fires when Energy decreases or slightly increases
        assert Energy - Energy_prev < E_epsilon, "Energy > Energy_prev??"

        #update best node, if energy decreased
        if Energy - Energy_prev < 0:
            optim_node = curr_node

        #update current node
        new_node = find_new_branch(K_tree, curr_node, K_path)
        if type(new_node)!=tuple:
            termination = "Whole tree has been explored"
            break
        else:
            curr_node = new_node

        #update previous energy
        Energy_prev = Energy

    #ouput
    out, E_a = output(K_tree, optim_node, termination, matrix_min,ansatz_m,N,H_m, log, HVA)

    #Ratio between number of nfev theta_init to theta_t and theta_a to theta_t
    theta_a = out['angles']
    K_a = out["K"]
    appr_min = scipy.optimize.minimize(Energy_matrix, theta_a, jac = False, args = (N,H_m,ansatz_m, K_a, HVA))


    nfev_ratio = matrix_min.nfev/appr_min.nfev
    E_a_t = appr_min.fun #
    E_t = matrix_min.fun
    Overlap = overlap(matrix_min.x, theta_a, K_init, K_a, ansatz_m, HVA = HVA)

    if log:
         print(f"{'Overlap <Ψ_t|Ψ_a>:':<25} {f'{Overlap}'}\n")
    print(f"nfev_t/nfev_a: {matrix_min.nfev}/{appr_min.nfev} = {matrix_min.nfev/appr_min.nfev}")
    
    return nfev_ratio, Overlap, E_a, E_t, E_a_t, matrix_min



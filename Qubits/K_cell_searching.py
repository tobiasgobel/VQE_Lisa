from Func import *
from deprecated import *
from s_dict_builder import *
from custom_optimizer import *
from pauli_objects import *
from Energy_funcions import *
from Ansatzes_Hamiltonians import *
from cirq_energy import *
from tqdm import tqdm
from visualize_landscape import *

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
        reduced_K_i = [k%4 for k in K_i]
        if not K_tree[tree_i]['seen'] and list(reduced_K_i) not in K_path:
            updated = True
            return tree_i

    if not updated:
        return find_new_branch(K_tree, node_up, K_path)

def output(K_tree, optim_node, termination, matrix_min, ansatz, N, H, log = True, HVA = False, method = "SLSQP"):
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
def find_K(N, ansatz, H, iterations, order, boundary = "hypersphere",log=True, matrix_min = None, HVA = False, method = "BFGS"):

    #convert to cirq objects
    H_cirq = sum([h.cirq_repr() for h in H])
    ansatz_cirq = [a.cirq_repr() for a in ansatz]


    N_K = iterations #number of iterations

     #random initial K-cell
    if HVA:
        theta_init = HVA_initialisation(HVA, 3*np.pi/16)
        K_init = [0]*HVA
        K_init = distribute_over_gates(HVA, N, K_init)
    else:
        theta_init = [1/np.pi/16]*len(ansatz)
        K_init= [0]*len(ansatz)

    
    K = K_init.copy()

    K_tree = {(0,):{"K":K,"angles":theta_init},(0,0):{"K":K, "seen":False}}
    Energy_prev = np.inf
    termination = "Maximum number of iters has been reached"
    curr_node = (0,0)
    optim_node = (0,0) #node with lowest energy
    epsilon = 1e-3 # angle close enough to magic border of pi/8
    E_epsilon = 1e-5 # Energy must not increase with more than this value
    K_path = [] #store all K's visited modulo 4

    #calculate minimum
    start = time()
    if matrix_min is None:
        matrix_min = scipy.optimize.minimize(cirq_Energy, theta_init, args = (N, ansatz_cirq, H_cirq, K, HVA),method = method)
        print(matrix_min.fun)
    end = time()
    print(f"{'Time to find local minimum:':<25} {f'{end-start}'}\n")
    if log: print("LOG")

    for iter in tqdm(range(N_K)):

        K = K_tree[curr_node]["K"]
        s = s_dict(N, ansatz, K, order)
        G_K = G_k(N, H, ansatz,K)
        sdicts, lc = s_dicts_lightcones(N, ansatz, H, K, order)

        #previous minimized set of angles
        node_above= K_tree[curr_node[:-1]]


        #calculate energy
        if HVA:
            args = (s, G_K, order, HVA)
            prev_angles_global = local_to_global_angle(node_above["angles"], shorten_vector(HVA, N, node_above["K"]))
            init_angles = global_to_local_angle(prev_angles_global, shorten_vector(HVA, N, K))
        else:
            #args = (s, G_K, order, HVA) without lightcones

            #with lightcones
            args = (sdicts, G_K, lc, order)


            prev_angles_global = local_to_global_angle(node_above["angles"], node_above["K"])
            #translate to angles in current K_cell
            init_angles = np.array(global_to_local_angle(prev_angles_global, K).copy())

        optimizer = E_optimizer(energy_lightcone, init_angles, method = method, args = args)
        result = optimizer.optim()

        #get indices of magic gates
        gates_zipped = zip(np.pi/8 - abs(result.x), range(len(result.x)))

        #sort by distance to magic angle
        gates_sorted = sorted(gates_zipped)

        #get indices of gates close enough to magic angle
        magic_indices = [gates_sorted[1] for gates_sorted in gates_sorted if gates_sorted[0] < epsilon]
        print('Magic indices: ', magic_indices)



        N_magic = len(magic_indices)
        Energy = result.fun


        #add data to dictionary
        K_tree[curr_node]["seen"] = True
        K_tree[curr_node]["N_magic"]= N_magic
        K_tree[curr_node]["energy"] = Energy
        K_tree[curr_node]["angles"] = result.x

        #add K-vector corresponding to curr_node to K_path
        K_path += [([k%4 for k in list(K).copy()])]

        if log:
            print("-"*30)
            print(f"iteration: {iter}")
            for key, value in K_tree[curr_node].items():
                print(f"{f'{key}:':<25}{f'{value}'}")

        #if Energy increases, find new branch skip iteration
        if Energy - Energy_prev > E_epsilon:
            new_node = find_new_branch(K_tree, curr_node[:-1], K_path)
            print(curr_node, Energy, shorten_vector(HVA, N, K_tree[curr_node]["K"]))
            if type(new_node)!=tuple:
                termination = "Whole tree has been explored"
                break
            else: 
                curr_node = new_node
                continue
        

        #include new branches
        for i in range(N_magic):
            magic_index = magic_indices[i]
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
        print(curr_node, Energy, shorten_vector(HVA, N, K_tree[curr_node]["K"]))
        if type(new_node)!=tuple:
            termination = "Whole tree has been explored"
            break
        else:
            curr_node = new_node

        #update previous energy
        Energy_prev = Energy
        


    ansatz_m = [a.matrix_repr() for a in ansatz]
    H_m = sum([h.matrix_repr() for h in H])
    #ouput
    if log:
        output(K_tree, optim_node, termination, matrix_min,ansatz_m,N,H_m, log, HVA)

    out = K_tree[optim_node]
    E_a = cirq_Energy(out["angles"], N, ansatz_cirq, H_cirq, out["K"], HVA)

    #Ratio between number of nfev theta_init to theta_t and theta_a to theta_t
    theta_a = out['angles']
    K_a = out["K"]
    #landscape_visualize(theta_a, cirq_Energy, (N, ansatz_cirq, H_cirq, K_a, HVA), filename= "landscape.png", num_directions = len(theta_a))
    # appr_min = scipy.optimize.minimize(cirq_Energy, theta_a, jac = False, args = (N, ansatz_cirq, H_cirq, K_a, HVA), method = 'COBYLA', options = {'rhobeg':0.01})
    appr_min = E_optim_cirq(cirq_Energy, theta_a, args = (N, ansatz_cirq, H_cirq, K_a, HVA), method = method).optim()
    
    nfev_ratio = matrix_min.nfev/appr_min.nfev
    E_a_t = appr_min.fun #
    E_t = matrix_min.fun
    Overlap = overlap(matrix_min.x, theta_a, K_init, K_a, ansatz_m, HVA = HVA)
    if log:
         print(f"{'Overlap <Ψ_t|Ψ_a>:':<25} {f'{Overlap}'}\n")
    print(f"nfev_t/nfev_a: {matrix_min.nfev}/{appr_min.nfev} = {matrix_min.nfev/appr_min.nfev}")
    
    return nfev_ratio, Overlap, E_a, E_t, E_a_t, matrix_min




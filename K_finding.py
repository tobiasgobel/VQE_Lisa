from VQE_functions import *
from tqdm import tqdm
N = 7
L = 2
X_H = -1
ansatz = QAOA(N, L)
H = TFIM(N, X_H)
iterations = 10
order = 6
boundary = "hypersphere"
filename = "result_optim.csv"
matrix_min = None
input = [N, "QAOA", "TFIM", X_H, L, order, boundary, iterations]

for _ in tqdm(range(20)):

    print(input)
    nfev_ratio, Overlap, E_a, E_t, E_a_t,_ = find_K(N, ansatz, H, iterations, order, log = False, matrix_min = matrix_min)
    output = [nfev_ratio, Overlap, E_a, E_t, E_a_t]

    line = input + output 
    print(line)

    from csv import writer
    with open(filename, 'a') as f_object:
    
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(line)
    
        # Close the file object
        f_object.close()

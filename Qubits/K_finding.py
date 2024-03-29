from K_cell_searching import *
from tqdm import tqdm

for N in [6]:
    L = 2
    HVA = False
    X_H = -1
    ansatz = QAOA(N, 4)
    H = TFIM(N, X_h = 1)
    iterations = 10
    order = 4
    boundary = "hypersphere"
    filename = "table.csv"
    matrix_min = None
    input = [N, "QAOA", "TFIM", X_H, L, order, boundary, iterations]
    methods = ["SLSQP"]
    for method in methods:
        start = time()
        print(f"method: {method}")

        print(input)
        nfev_ratio, Overlap, E_a, E_t, E_a_t,_ = find_K(N, ansatz, H, iterations, order, log = True, matrix_min = matrix_min, HVA =HVA, method = method)
        output = [nfev_ratio, Overlap, E_a, E_t, E_a_t]
        #round numbers to 3 decimals
        output  = [round(i, 3) for i in output]

        line = input + [method] + output + [np.round(time() - start, 1)]
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

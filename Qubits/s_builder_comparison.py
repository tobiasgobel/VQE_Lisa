from K_cell_searching import *
from cirq_energy import *
from time import time
from visualize_landscape import *
from sys import getsizeof


def count_number_of_terms(s):
    n_terms =0
    for st in s:
        lst = s[st]
        n_terms += len(np.array(lst[1]))
    return n_terms

def tan_prd_list(s, thetas):
    tan_prds = []
    tan_thetas = np.tan(thetas)
    for st in s:
        lst = s[st]
        for act in lst[0]:
            tan_prds.append(tangent_product(act, tan_thetas))
    tan_prds.sort()
    return tan_prds



def plot_activation_distribution(ansatz_len, s, filename):
    gates = [f"Gate {i}" for i in range(ansatz_len)]
    term_counts = {}
    for st in s:
        lst = s[st]
        orders = np.sum(lst[0], axis = 1)
        for i in range(len(lst[1])):
            act = lst[0][i,:]
            order = orders[i]
            if order not in term_counts:
                term_counts[order] = act
            else:
                term_counts[order] += act
    print(term_counts)

    width = 0.6  # the width of the bars: can also be len(x) sequence


    fig, ax = plt.subplots()
    bottom = np.zeros(ansatz_len)

    for order, order_count in term_counts.items():
        p = ax.bar(gates, order_count, width, label=f"order {order}", bottom=bottom)
        bottom += order_count

    ax.set_title('Number of terms per gate by order')
    ax.set_xticklabels(gates, rotation=90)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.savefig(filename)
    plt.show()




N = 8
H = TFIM(N, -1)#Z_expectation_val(N,array_method = False)
ansatz_full = QAOA(N, 8)#random_circuit(N, 8, array_method=False)
lightc = lightcone(H, ansatz_full, order_max = 5)

ansatz = lightc[5]
print(len(ansatz), len(lightc[2]),len(ansatz_full))

thetas = np.random.randn(len(ansatz))*0.3
print("thetas", thetas)
K = np.random.choice([0,-1, 1, 2, -2], len(ansatz))
# print('-----------')
# print('Perturabation theory s -tree')
# order = 3
# time_expansion = time()
# s = s_dict(N, ansatz, K, order)
# time_expansion = time() - time_expansion
# print("number of terms:",count_number_of_terms(s))
# print("time for s_dict:", time_expansion)
# #plot_activation_distribution(len(ansatz), s, "s_histogram.png")

# print('-----------')
# print('Angle based s -tree ')
# time_s = time()
# s_angle = s_dict_tree(N, ansatz, K, thetas, treshold = 1e-2)
# time_s = time() - time_s
# print("number of terms:",count_number_of_terms(s_angle))
# print("time for s_dict:", time_s)


# n_terms_1 = []
# times_1 = []
# for order in [3,4,5,6,7]:
#     print('order', order)
#     time_expansion = time()
#     s = s_dict(N, ansatz, K, order)
#     time_expansion = time() - time_expansion
#     n_terms = count_number_of_terms(s)
#     n_terms_1.append(n_terms)
#     times_1.append(time_expansion)
#     print("number of terms:",count_number_of_terms(s))
#     print("time for s_dict:", time_expansion)



# n_terms_2 = []
# times_2 = []
# for treshold in [0.1,0.07,0.05,0.01,5e-3,5e-4,1e-4]:
#     print('treshold', treshold)
#     time_s = time()
#     s_angle = s_dict_tree(N, ansatz, K, thetas, treshold = treshold)
#     time_s = time() - time_s
#     n_terms2 = count_number_of_terms(s_angle)
#     n_terms_2.append(n_terms2)
#     times_2.append(time_s)
#     print("number of terms:",n_terms2)
#     print("time for s_dict:", time_s)

# plt.plot(n_terms_2, times_2, label = "treshold")
# plt.plot(n_terms_1, times_1, label="perturbative")
# plt.legend()
# plt.xlabel("Number of terms")
# plt.ylabel("Time (s)")
# plt.savefig("time_complexities.png")




import pandas as pd
samples = 15
sigmas = [0.08,0.16,0.32,0.4]
orders = [3,4,5,6,7,8]
tresholds = np.arange(samples)
data = pd.DataFrame(columns = sigmas, index = orders)
data_2 = pd.DataFrame(columns = sigmas, index = tresholds)
n_terms_1 = pd.DataFrame(columns = sigmas, index = orders)
n_terms_2 = pd.DataFrame(columns = sigmas, index = tresholds)

for sigma in sigmas:
    First_time = True
    thetas = np.random.randn(len(ansatz))*sigma
    K = np.random.choice([0,1,-1,2,-2], len(ansatz))

    H_cirq = sum([h.cirq_repr() for h in H])
    ansatz_cirq = [a.cirq_repr() for a in ansatz]
    E_cirq = cirq_Energy(thetas, N, ansatz_cirq, H_cirq, K)


    for order in orders:
        print(f"delta_theta: {sigma} and order: {order}")
        s = s_dict(N, ansatz, K, order)
        G_K = G_k(N, H, ansatz, K)
        E_expansion = energy(thetas, s, G_K, order)
        data[sigma][order] = np.log(np.abs(E_expansion-E_cirq))
        n_terms_1[sigma][order] = count_number_of_terms(s)

    tangent_products = tan_prd_list(s, thetas)
    trsh = np.linspace(0, len(tangent_products)-1,samples+5, dtype =int)
    trsh = [tangent_products[i] for i in trsh[5:]]

    for j, tresh in enumerate(tresholds):
        s_angle = s_dict_tree(N, ansatz, K, thetas, treshold = trsh[j])
        G_K = G_k(N, H, ansatz, K)
        # print(s_angle)
        try:
            E_expansion = energy(thetas, s_angle, G_K, order)
        except:
            E_expansion = 0

        data_2[sigma][tresh] = np.log(np.abs(E_expansion-E_cirq))
        n_terms_2[sigma][tresh] = count_number_of_terms(s_angle)



data.to_csv("reproducing_results/data/s_comp_1.csv", sep = ";")
data_2.to_csv("reproducing_results/data/s_comp_2.csv", sep = ";")

a = pd.read_csv("reproducing_results/data/s_comp_1.csv", index_col = 0, delimiter = ";")
b = pd.read_csv("reproducing_results/data/s_comp_2.csv",index_col = 0, delimiter = ";")

fig, axs = plt.subplots(2, 2, sharey=True)


axs = axs.flatten()
for k, sigma in enumerate(a.columns):
    l_1 = [i for i in a[sigma]]
    l_2 = [i for i in b[sigma]]
    nterms = n_terms_1.values[:,k]
    nterms2 = n_terms_2.values[:,k]
    axs[k].plot(nterms, l_1, label = f"Perturbation")
    axs[k].plot(nterms2, l_2, label = "Angle dependent")
    axs[k].set_title(f"Sigma {sigma}")
    axs[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.text(0.5, 0.01, 'Number of terms', ha='center')
fig.text(0.01, 0.5, "log|Error|", va='center', rotation='vertical')
fig.tight_layout()
axs[1].legend()

fig.savefig('s_comparison_2.png')   # save the figure to file
plt.close(fig) 
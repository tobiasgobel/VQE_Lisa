import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("reproducing_results/data/repr_3.csv", index_col = 0, delimiter = ",")
a.to_csv("reproducing_results/data/repr_3.csv", sep = ";")
a = pd.read_csv("reproducing_results/data/repr_3.csv", index_col = 0, delimiter = ";")

#plot energy for different orders
# pick only first entry
#convert strings to tuples
orders = [3,4,5,6,7,8,9,10,11,12,13,14,15]
# for delta in a.columns:
#     l = [abs(eval(i)[0]-eval(i)[1]) for i in a[delta]]
#     plt.plot(orders, l, label = f"delta theta {delta}")
# plt.legend()
# plt.xlabel("Order K")
# plt.ylabel("|<Z1Z5> - <Z1Z5>_cirq|")
# plt.savefig("reproducing_results/reproduced_graphs/theta_comp_2.png")
# plt.show()

# for delta in a.columns:
#     l = [np.log(abs(eval(i)[0]-eval(i)[1])) for i in a[delta]]
#     plt.plot(orders, l, label = f"delta theta {delta}")
# plt.legend()
# plt.xlabel("Order K")
# plt.ylabel("log|<Z1Z5> - <Z1Z5>_cirq|")
# plt.savefig("reproducing_results/reproduced_graphs/theta_comp_log_2.png")

for order in orders:
    line = [abs(eval(i)[0]) for i in a.loc[order]]
    plt.plot(a.loc[order].index, line, label = f"order {order}")

plt.legend()

plt.title("Random 4 layer ansatz with 20 qubits ")
plt.xlabel("delta theta")
plt.ylabel("Energy <Z1Z10>")
plt.savefig("reproducing_results/reproduced_graphs/order_comparison_3.png")
    


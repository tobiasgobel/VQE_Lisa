import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("reproducing_results/data/repr_3.csv", index_col = 0, delimiter = ";")
print(a)
#plot energy for different orders
# pick only first entry
#convert strings to tuples
orders = [3,4,5,6,7,8]
# for delta in a.columns:
#     l = [abs(eval(i)[0]-eval(i)[1]) for i in a[delta]]
#     plt.plot(orders, l, label = f"delta theta {delta}")
# plt.legend()
# plt.xlabel("Order K")
# plt.ylabel("|<Z1Z5> - <Z1Z5>_cirq|")
# plt.savefig("reproducing_results/reproduced_graphs/theta_comp_2.png")
# plt.show()
k = 0
for delta in a.columns:
    k+=1
    if k%3 == 0:
        continue
    l = [np.log(abs(eval(i)[0]-eval(i)[1])) for i in a[delta]]
    plt.plot(orders, l, label = f"delta theta {delta}")
plt.legend()
plt.xlabel("Order K")
plt.ylabel("log|<Z1Z5> - <Z1Z5>_cirq|")
plt.savefig("reproducing_results/reproduced_graphs/theta_comp_log_3.png")

# for order in orders:
#     line = [abs(eval(i)[0]) for i in a.loc[order]]
#     indices = [eval(i) for i in a.loc[order].index]
#     plt.plot(indices, line, label = f"order {order}")

# plt.legend()
# print(plt.xticks())
# plt.xticks()
# plt.title("Random 4 layer ansatz 50 qubits")
# plt.xlabel("delta theta")
# plt.ylabel("Energy <Z1Z10>")
# plt.savefig("reproducing_results/reproduced_graphs/order_comparison_3.png")
    


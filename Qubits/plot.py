from K_cell_searching import *
import pandas as pd
#plot histogram of nfev_ratio
filename = "VQE/results.csv"
df = pd.read_csv(filename)


class visualize:
    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
        print(self.kwargs)

    @property
    def data(self):
        #select data from dataframe
        for key, value in self.kwargs.items():
            self.df = self.df.loc[self.df[key] == value]
        return self.df

    def get_mean(self, column):
        return self.data.loc[:,column].mean()

    def get_std(self, column):
        return self.data.loc[:,column].std()

    #percentage of times overlap is close to 1
    def overlap_ratio(self,epsilon = 0.1):
        overlap = self.data.loc[:,"overlap"]
        assert overlap.count() != 0, "No data found"
        overlap_ratio = overlap[overlap > 1-epsilon].count()/overlap.count()
        return overlap_ratio


    
    # def plot_nfev_ratio(self):
    #     #get column pandas nfev_ratio
    #     nfev_ratio = self.data.loc[:,"nfev_ratio"]
    #     #plot histogram
    #     mean = self.get_mean("nfev_ratio")
    #     std = self.get_std("nfev_ratio")
    #     nfev_ratio.hist(bins = 40,density = True)
    #     overlap_ratio = self.overlap_ratio()
    #     #vertical line
    #     plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label = f"mean = {mean:.2f}")
    #     plt.plot([],[],'', label = f"std = {std:.2f}")
    #     plt.plot([],[],'', label = f"overlap ratio = {overlap_ratio:.2f}")

    #     plt.plot()
    #     plt.xlabel("nfev_ratio")
    #     plt.ylabel("frequency")
    #     plt.title(f"args = {self.kwargs}")
    #     plt.legend()
    #     plt.savefig(f"plots/QAOA/TFIM/nfev_ratio/{self.kwargs}.png")




# hypercube = visualize(df, boundary = "hypercube",X_h=-0.1)
hypersphere = visualize(df, N = 6, L = 2, boundary = "hypersphere", X_h = -0.1, ansatz = "QAOA", iterations = 10, order = 6)

print(len(hypersphere.data))
# hypersphere.plot_nfev_ratio()

# hypersphere_overlap_ratio = hypersphere.overlap_ratio()
# hypercube_overlap_ratio =  hypercube.overlap_ratio()

# a = visualize(df, N=4, L=2, X_h=-1, order=6, boundary="hypercube").data
# b = visualize(df, N=4, L=2, X_h=-1, order=6, boundary="hypersphere").data

# #compare histograms
# a = a.loc[:,"nfev_ratio"]
# b = b.loc[:,"nfev_ratio"]

# #mean
# a_mean = a.mean()
# b_mean = b.mean()

# #std
# a_std = a.std()
# b_std = b.std()

# #plot vertical lines
# plt.axvline(x=a_mean, color='r', linestyle='dashed', linewidth=1, label = f"mean hypercube {np.round(a_mean,2)}")
# plt.axvline(x=b_mean, color='b', linestyle='dashed', linewidth=1, label = f"mean hypersphere {np.round(b_mean,2)}")

# plt.hist(a, bins = 20, alpha = 0.5, label = "hypercube",density=True)
# plt.hist(b, bins = 20, alpha = 0.5, label = "hypersphere",density=True)

# plt.plot([], [], ' ', label="overlap ratio hypercube = " + str(np.round(hypercube_overlap_ratio,2)))
# plt.plot([], [], ' ', label="overlap ratio hypersphere = " + str(np.round(hypersphere_overlap_ratio,2)))

# plt.xlabel("nfev_ratio")
# plt.ylabel("frequency")
# plt.title("TFIM, QAOA, N = 4, L = 2, X_H = -1, order = 6")
# plt.legend()
# plt.savefig("plots/Manual/histograms_N=4_L=2_X_H=-1_order=6.png")





#plot nfev ratio for N = 6 and N = 4 histogram
# N4 = visualize(df, N = 4, L = 2, boundary = "hypersphere", X_h = -10, ansatz = "QAOA", iterations = 10, order = 6).data
# N5 = visualize(df, N = 5, L = 2, boundary = "hypersphere", X_h = -10, ansatz = "QAOA", iterations = 10, order = 6).data
# N6 = visualize(df, N = 6, L = 2, boundary = "hypersphere", X_h = -10, ansatz = "QAOA", iterations = 10, order = 6).data

# a = N4.loc[:,"nfev_ratio"]
# b = N6.loc[:, "nfev_ratio"]
# c = N5.loc[:,"nfev_ratio"]

# #mean
# a_mean = a.mean()
# b_mean = b.mean()
# c_mean = c.mean()

# #std
# a_std = a.std()
# b_std = b.std()
# c_std = c.std()

# #plot vertical lines
# plt.axvline(x=a_mean, color='r', linestyle='dashed', linewidth=1, label = f"mean N:4 {np.round(a_mean,2)}")
# plt.axvline(x=b_mean, color='b', linestyle='dashed', linewidth=1, label = f"mean N:6 {np.round(b_mean,2)}")
# plt.axvline(x=c_mean, color='g', linestyle='dashed', linewidth=1, label = f"mean N:5 {np.round(c_mean,2)}")

# plt.hist(a, bins = 20, alpha = 0.5, label = "N:4",density=True)
# plt.hist(c, bins = 20, alpha = 0.5, label = "N:5",density=True)
# plt.hist(b, bins = 20, alpha = 0.5, label = "N:6",density=True)

# plt.legend()
# plt.title("nfev-ratio histograms for N:4,5,6")
# plt.savefig("VQE/plots/Manual/histograms_N=4&6_L=2_X_H=-10_order=6.png")


#plot nfev ratio for N=6 with overlap > 0.95
N6_overlap = visualize(df, N = 7, L = 2, boundary = "hypersphere", X_h = -10, ansatz = "QAOA", iterations = 10, order = 6).data
print(N6_overlap)
N6_overlap = N6_overlap.loc[N6_overlap["overlap"] < 0.95, :]    
N6_overlap = N6_overlap.loc[:,"nfev_ratio"]

print(N6_overlap.mean())






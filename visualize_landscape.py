import matplotlib as plt
import numpy as np
from K_cell_searching import *

def explore(point, index, r):
    theta = point.copy()
    theta[index] +=r
    return theta

def landscape_visualize(point, E_function, args, num_directions = 5, scale = .5, filename ="tesplot.png"):

    sweep = np.linspace(-scale, scale)
    sweep -= min(abs(sweep))
    print("minimum abs", min(abs(sweep)))

    for i in range(num_directions):
        E = [E_function(explore(point, i, r), *args) for r in sweep]
        plt.plot(sweep, E, label = f"theta_{i}")

    plt.title(f"Landscape around point {E_function.__name__}")
    plt.xlabel("deviation (radians)")
    plt.ylabel("Energy")
    plt.savefig("landscape_plots/"+filename)



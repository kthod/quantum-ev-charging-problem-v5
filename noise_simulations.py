from qiskit import *
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize
from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import time
from VQE_Class import*
from CompressedVQE_Class import*
from CompressedVQE_RPA_Class import*
from ClusterVQE_Class import*
from ClusterCompressedVQE_Class import*
import matplotlib.colors as mcolors
from quantum_MPC import *



epsilon = np.array(1*[26880,26880,26880,7680,23040,23040,7680,7680])/1000
de = 1*np.array(1*[3,3,3,1,3,3,1,1])
evs = 2
V=0.240
DT = 1
Horizon = 4


layers = [2,4,6]
F = [(0.001,0.01),(0.01,0.01),(0.01,0.1)]
markers = ['*','^','o']
min_cost = np.zeros((len(layers),5))
full_cost = np.zeros((len(layers),5))
for j in range(len(markers)+1):
    for i in range(len(layers)):




        inst_min = Quantum_MPC(epsilon = epsilon , de = de,C =100, Horizon = Horizon, DT = DT, layers=layers[i])
        

        label_min = r'Compressed VQE'
        marker_min = 's'
        # label_full = r'VQE'
        if (j>0):
            inst_min.add_noise(t1 = 50e3, t2 = 70e3, prob_1_qubit = F[j-1][0] ,prob_2_qubit = F[j-1][1], p1_0 = 0.1, p0_1 = 0.1)
            label_min = f'Compressed VQE $F_1 = {100-10*F[j-1][0]}\%, F_2 = {100-10*F[j-1][1]}\%$'
            marker_min =  markers[j-1]


        for k in range(5):

            inst_min.optimize(n_measurements = 10000, number_of_experiments = 1, maxiter = 500)
            #inst_min.optimize(n_measurements = 10000,number_of_experiments = 1,maxiter=500)


            min_cost[i,k] = inst_min.get_optimal_cost()



        

    upper_bound_min = [np.max(min_cost[i, :]) for i in range(len(layers))]
    lower_bound_min = [np.min(min_cost[i, :]) for i in range(len(layers))]
    mean_min = [np.mean(min_cost[i, :]) for i in range(len(layers))]


    plt.fill_between(layers, upper_bound_min, lower_bound_min, alpha=0.4)
    plt.plot(layers, mean_min, marker=marker_min, linestyle='--', label=label_min)
    # plt.fill_between(range(3), upper_bound_full, lower_bound_full, alpha=0.4)
    # plt.plot(range(3), mean_full, marker='^', linestyle='--', label=label_full)

plt.xlabel("Layers L")
plt.ylabel("Cost Function")
plt.title("Noise Simulation")
plt.legend()
plt.show()





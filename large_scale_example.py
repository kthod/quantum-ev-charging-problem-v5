from qiskit import *

import numpy as np

import numpy as np

import matplotlib.pyplot as plt
from VQE_Class import*
from CompressedVQE_Class import*
from CompressedVQE_RPA_Class import*
from ClusterVQE_Class import*
from ClusterCompressedVQE_Class import*
import matplotlib.colors as mcolors
from quantum_MPC import Quantum_MPC

epsilon = np.array(8*[26880,26880,26880,7680,23040,23040,7680,7680])/1000
de = 2*np.array(8*[3,3,3,1,3,3,1,1])

# epsilon = np.array([26880,26880,26880,7680])/1000
# de = np.array([3,3,3,1])

evs = len(epsilon)
V=0.240
DT = 0.5
Horizon = 12

num_of_experiments = 5
#-7414764.351285682
inst_min = Quantum_MPC(epsilon = epsilon , de = de,C =100, Horizon = Horizon, DT = DT, layers=2)
inst_min.optimize(n_measurements = 5000, number_of_experiments = num_of_experiments, maxiter = 200)
opt_val = inst_min.get_optimal_cost()
# print(opt_val)
shots = [1000,10000,100000]
lay = [2,4,6]
colorss = ["violet", "blueviolet","lightsteelblue"]
colorsl = ["mediumturquoise","mediumseagreen" ,"greenyellow"]
instances = []
for i in range(len(lay)):


    inst_min = Quantum_MPC(epsilon = epsilon , de = de,C =100, Horizon = Horizon, DT = DT, layers=2)
    inst_min.optimize(n_measurements = 10000, number_of_experiments = num_of_experiments, maxiter = 200)
    print(inst_min.get_sched())
    inst_min.plot_evolution(normalization=[opt_val], label=f"Compressed VQE Layers = {lay[i]}", color = colorsl[i])
    instances.append(inst_min)

plt.title(f"Evaluation of Cost Function for {evs} EV {Horizon} timesteps", fontsize=16)
plt.ylabel('Cost Function', fontsize=14)
plt.xlabel('Iteration', fontsize=14)

plt.legend( fontsize=12)
plt.show()


solutions =10

#scatter = inst.get_solution_distribution()

markers = ['+', 'x', '.']
for i in range(len(lay)):

    scatter = instances[i].get_solution_distribution(normalization=[opt_val], label=f"Shots = {shots[i]}", marker = markers[i], solutions = 10, shots = 10000)

norm = mcolors.Normalize(vmin=0, vmax=1)
cb = plt.colorbar(mappable=scatter, norm=norm)
cb.set_label('Fraction of solutions', fontsize=12)
plt.yticks(range(1,num_of_experiments+1))
plt.xlabel('Cost Function', fontsize=14)
plt.ylabel('Optimization run', fontsize=14)
plt.title(f"Distribution of solutions for {evs} EV {Horizon} timesteps", fontsize=16)
plt.grid(True)
plt.legend( fontsize=12)
plt.show()


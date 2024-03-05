
# Import necessary libraries
from qiskit import *  # Import everything from Qiskit for quantum computing
import numpy as np  # Import NumPy for numerical operations
import networkx as nx  # Import NetworkX for graph operations (not used in the shown code)
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib.colors as mcolors  # Import Matplotlib colors for color operations
from quantum_MPC import Quantum_MPC  # Import the Quantum_MPC class for quantum model predictive control

# Define epsilon and de arrays with specific values, representing parameters for the Quantum_MPC
epsilon = np.array([26880,26880,26880,7680])/1000
de = np.array([3,3,3,1])

# Define various parameters for the Quantum MPC setup
evs = len(epsilon)  # Number of electric vehicles or elements in epsilon

DT = 1  # Delta time, the time step for the prediction horizon
Horizon = 4  # The prediction horizon, i.e., how far into the future the MPC should optimize

# Define experiment parameters
shots = [1000,10000,50000]  # Number of measurements for quantum experiments
lay = [2,4,6]  # Different layer configurations for the quantum circuits
number_of_experiments = 10  # Number of optimization experiments to run

C = 12*16*240/1000  

# Initialize an instance of Quantum_MPC with specific parameters for compressed VQE layer configuration
inst_min = Quantum_MPC(epsilon=epsilon, de=de, C=C, Horizon=Horizon, DT=DT, layers=2)

# Optimize the Quantum MPC instance with specified parameters
inst_min.optimize(n_measurements=10000, number_of_experiments=number_of_experiments, maxiter=300)
print(inst_min.get_sched())  # Print the optimal schedule found by the optimization
inst_min.plot_evolution(normalization=[inst_min.get_optimal_cost()])  # Plot the cost function evolution over iterations

# Initialize another Quantum_MPC instance for a full encoding using the VQE algorithm
inst_full = Quantum_MPC(epsilon=epsilon, de=de, C=C, Horizon=Horizon, DT=DT, algorithm="VQE", layers=2)
inst_full.optimize(n_measurements=50, number_of_experiments=number_of_experiments, maxiter=300)
print(inst_full.get_sched())  

inst_full.plot_evolution(normalization=[inst_min.get_optimal_cost()], color='C2')

# Set up the plot for evaluating the cost function over the optimization iterations
plt.title(f"Evaluation of Cost Function for {evs} EV {Horizon} timesteps", fontsize=16)
plt.ylabel('Cost Function', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Plot the distribution of solutions for a given number of top solutions and shots
scatter = inst_full.get_solution_distribution(normalization=[inst_min.get_optimal_cost()], solutions=10, shots=1000)
scatter = inst_min.get_solution_distribution(normalization=[inst_min.get_optimal_cost()], solutions=10, shots=10000)
norm = mcolors.Normalize(vmin=0, vmax=1)
plt.colorbar(mappable=scatter, norm=norm, label='Fraction of solutions')
plt.yticks(range(1, number_of_experiments+1))
plt.xlabel('Cost Function', fontsize=14)
plt.ylabel('Optimization run', fontsize=14)
plt.title(f"Distribution of solutions for {evs} EV {Horizon} timesteps", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()



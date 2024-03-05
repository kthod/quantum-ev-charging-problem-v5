from qiskit import *
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError, QuantumError
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractclassmethod
import time
from scipy.optimize import minimize
class optimizationAlgorithm(ABC):
    """
    A base class for optimization algorithms using Quantum Approximate Optimization Algorithm (QAOA) or 
    Variational Quantum Eigensolver (VQE) approaches.
    
    This abstract base class defines the structure for optimization algorithms, including initialization,
    circuit generation, cost function evaluation, optimization process, and plotting the evolution of the cost function.
    
    Attributes:
    - qubo_matrix (np.ndarray): The QUBO matrix representing the problem to be solved.
    - layers (int): The number of layers in the quantum circuit.
    - nq (int): The number of qubits used in the quantum circuit.
    - algorithm (str): The name of the algorithm.
    - status (str): The current status of the algorithm instance.
    - optimal_answer (str): The optimal solution found by the algorithm.
    - optimal_cost (float): The cost associated with the optimal solution.
    - cost_evolution (list): A list to track the cost evolution over optimization iterations.
    
    Abstract Methods:
    - circuit: Should return a QuantumCircuit based on input parameters.
    - cost_function: Defines the cost function to be minimized.
    - optimize: Implements the optimization process.
    """

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1) -> None:
        """
        Initializes the optimization algorithm with a given QUBO matrix and number of layers for the quantum circuit.
        
        Parameters:
        - qubo_matrix (np.ndarray): The QUBO matrix for the optimization problem.
        - layers (int, optional): The number of layers in the quantum circuit. Defaults to 1.
        """
        self.qubo_matrix = qubo_matrix
        self.layers = layers
        self.nq = 0  # This should be set based on the problem or circuit specifics in derived classes


        # Algorithm metadata
        self.algorithm = "-"
        self.status = "INITIALIZED"
        self.optimal_answer = "-"
        self.optimal_cost = 0
        self.noise_model = None
        # Cost evolution tracking
        self.cost_evolution = list()

    def __str__(self) -> str:
        """
        Returns a string representation of the optimization algorithm instance, including its status and results.
        """
        output = (
            "======================== <ANGELQ OPT ALGORITHM> ========================\n"
            "Number of classical variables:                      {}\n"
            "Number of qubits:                                   {}\n"
            "Algorithm:                                          {}\n\n"
            "Status:                                             {}\n"
            "Solution:                                           {}\n"
            "Cost:                                               {}\n"
            "========================================================================\n"
        ).format(len(self.qubo_matrix), self.nq, self.algorithm, self.status, self.optimal_answer, self.optimal_cost)

        return output
    
    @abstractclassmethod
    def circuit(self, theta: list) -> QuantumCircuit:
        """
        Abstract method for generating a quantum circuit based on the input parameters.
        
        Parameters:
        - theta (list): A list of parameters for the quantum circuit.
        
        Returns:
        - QuantumCircuit: The quantum circuit for the optimization algorithm.
        """
        raise NotImplementedError("Circuit generation not implemented.")
    
    @abstractclassmethod
    def cost_function(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
        """
        Abstract method for defining the cost function to be minimized by the optimization algorithm.
        
        Parameters:
        - maxiter (int, optional): The maximum number of iterations for the optimization. Defaults to 1000.
        - shots (int, optional): The number of measurement shots for each circuit execution. Defaults to 10000.
        - experiments (int, optional): The number of experiments to average over. Defaults to 1.
        
        Returns:
        - str: A string representation of the cost function evaluation result.
        """
        raise NotImplementedError("Cost function evaluation not implemented.")
    
    @abstractclassmethod
    def optimize(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
        """
        Abstract method for the optimization process, implementing how the algorithm seeks to find the optimal solution.
        
        Parameters:
        - maxiter (int, optional): The maximum number of iterations for the optimizer. Defaults to 1000.
        - shots (int, optional): The number of measurement shots for each circuit execution. Defaults to 10000.
        - experiments (int, optional): The number of experiments to perform. Defaults to 1.
        
        Returns:
        - str: A string indicating the outcome of the optimization process.
        """
        raise NotImplementedError("Optimization process not implemented.")
    
    def plot_evolution(self, normalization=[], label = "",color = "red") -> None:
        """
        Plots the evolution of the cost over the optimization iterations, showing the range and mean of the cost at each iteration.
        
        Optionally, normalizes the cost measurements to a specified range for visualization purposes.
        
        Parameters:
        - normalization (list, optional): Specifies the range [min, max] for normalizing the cost measurements. If not provided,
          no normalization is applied. The list must contain exactly two elements if provided.
        """
        assert len(normalization) == 0 or len(normalization) == 2 or len(normalization) ==1, "Normalization parameter must be empty or contain two or one  elements."
        if label == "":
            label = self.algorithm
        measurements = np.array(self.cost_evolution )
        maxiter = len(measurements[0])

        if len(normalization) == 2:
            vmin, vmax = normalization
            measurements = (measurements - np.array([[vmin] * maxiter] * len(measurements))) / np.array([[vmax - vmin] * maxiter] * len(measurements))
        if len(normalization) == 1:
            vmin = normalization
            measurements = (measurements - np.array([[vmin] * maxiter] * len(measurements))[:,1,:]) 

        upper_bound = [np.max(measurements[:, i]) for i in range(maxiter)]
        lower_bound = [np.min(measurements[:, i]) for i in range(maxiter)]
        mean = [np.mean(measurements[:, i]) for i in range(maxiter)]

        plt.fill_between(range(maxiter), upper_bound, lower_bound, color = color,alpha=0.5)
        plt.plot(range(maxiter), mean, linestyle='--', color = color,label=label)
        # plt.xlabel('Iteration')
        # plt.ylabel('Cost')
        # plt.legend()
        # plt.show()

    def add_noise(self,t1 = 50e3, t2 = 70e3, prob_1_qubit = 0.001 ,prob_2_qubit = 0.01, p1_0 = 0.2, p0_1 = 0.3):
        self.noise_model = NoiseModel()
        gate_time = 50  # Duration of the gate in nanoseconds
        

# Create thermal relaxation error
        thermal_relaxation_qubit1 = thermal_relaxation_error(t1, t2, gate_time)
        thermal_relaxation_qubit2 = thermal_relaxation_error(t1, t2, gate_time)

# Create a composite error for the two-qubit 'cx' gate
        thermal_relaxation_cx = QuantumError.tensor(thermal_relaxation_qubit2, thermal_relaxation_qubit1)

# Apply the composite thermal relaxation error to the 'cx' gate
        

        depolarizing_error_1_qubit = depolarizing_error(prob_1_qubit, 1)
        depolarizing_error_2_qubit = depolarizing_error(prob_2_qubit, 2)

        readout_error = ReadoutError([[1 - p0_1, p0_1], [p1_0, 1 - p1_0]])

        self.noise_model.add_all_qubit_quantum_error(depolarizing_error_1_qubit, ['u1', 'u2', 'u3'])
        self.noise_model.add_all_qubit_quantum_error(depolarizing_error_2_qubit, 'cx')

        # Add thermal relaxation errors to all qubits for 1 and 2 qubit gates
        self.noise_model.add_all_qubit_quantum_error(thermal_relaxation_qubit1, ['u1', 'u2', 'u3'])
        self.noise_model.add_all_qubit_quantum_error(thermal_relaxation_cx, 'cx')

        # Add readout error to all qubits
        self.noise_model.add_all_qubit_readout_error(readout_error)

    def clear_noise(self):
        self.noise_model = None
        
    def cost_function(self, parameters, number_of_measurements):
        """
        Evaluates the cost function for the given parameters by executing the quantum circuit and measuring the outcomes.
        
        This method constructs and executes a quantum circuit based on the provided parameters, then calculates the cost
        based on the measurement outcomes. The cost is determined by a problem-specific objective function, which should
        be implemented to reflect the optimization goal.
        
        Parameters:
        - parameters: The parameters to be used in the quantum circuit for this evaluation.
        - number_of_measurements: The number of measurements (shots) to be performed on the quantum circuit to estimate the cost.
        
        Returns:
        - obj: The evaluated cost for the given parameters.
        
        """
        # Construct the quantum circuit with the given parameters
        circ = self.circuit(parameters)

        # Execute the quantum circuit on a simulator with the specified number of shots
        job = execute(circ, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=number_of_measurements, max_parallel_threads=1)
        counts = job.result().get_counts(circ)

        # Calculate the objective value based on the counts
        obj = self.compute_expectation(counts)
        
        # Temporarily store the cost evolution for this optimization iteration
        self.temp_cost_evolution.append(obj)
        
        return obj

    def optimize(self, n_measurements=10000, number_of_experiments=10, maxiter=150):
        """
        Performs the optimization process to find the parameters that minimize the cost function.
        
        This method iteratively adjusts the parameters of the quantum circuit to minimize the cost function
        evaluated by `cost_function`. It uses a classical optimizer (e.g., COBYLA) to find the optimal parameters
        that lead to the minimum cost. The optimization is repeated across multiple experiments to ensure robustness
        and to explore the solution space thoroughly.
        
        Parameters:
        - n_measurements (int): The number of measurements (shots) for each execution of the quantum circuit.
        - number_of_experiments (int): The number of times the optimization process is repeated.
        - maxiter (int): The maximum number of iterations for the classical optimizer.
        
        """
        self.number_of_experiments = number_of_experiments
        sum_time = 0
        self.opt_vec = []  # Initialize list to store optimal function values
        self.x_vec = []    # Initialize list to store optimal parameters
        
        for k in range(number_of_experiments):
            self.temp_cost_evolution = []
            initial_vector = np.random.uniform(0, 2*np.pi, self.layers * self.nq)

            print(f"Begin optimization for hardware efficient {self.algorithm} and {n_measurements} measurements...")
            time1 = time.time()
            opt = minimize(self.cost_function, initial_vector, args=(n_measurements), method='COBYLA', options={'maxiter': maxiter})
            sum_time += time.time() - time1
            print(f"Optimization complete. Time taken: {time.time() - time1:.3f}s.\n")
            print(f"Final cost function value: {opt.fun:.5f}")
            print(f"Number of optimizer iterations: {opt.nfev}")

            self.cost_evolution.append(np.asarray(self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution))))

            self.opt_vec.append(opt.fun)
            self.x_vec.append(opt.x)

        ind = np.argmin(self.opt_vec)
        self.optimal_params = self.x_vec[ind]
        self.optimal_cost = np.min(self.opt_vec)
        self.status = "OPTIMIZED"


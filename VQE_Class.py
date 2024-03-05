from qiskit import *
from optimizationAlgorithm import optimizationAlgorithm
import numpy as np
import matplotlib.pyplot as plt

class VQE(optimizationAlgorithm):
    """
    A class representing the Variational Quantum Eigensolver (VQE) algorithm, which is a hybrid quantum-classical
    algorithm used to find the ground state energy of a Hamiltonian. This class extends the optimizationAlgorithm
    abstract base class, implementing the required quantum circuit construction, cost function computation,
    and optimization process specific to VQE.
    
    Attributes:
    - qubo_matrix (np.ndarray): The QUBO matrix representing the problem Hamiltonian.
    - layers (int): The number of layers in the variational quantum circuit.
    - nc (int): The number of classical variables, equivalent to the size of the QUBO matrix.
    - nq (int): The number of qubits used in the quantum circuit, equal to the number of classical variables.
    - algorithm (str): The name of the algorithm, set to "VQE" for instances of this class.
    - optimal_params (str or np.ndarray): The parameters that minimize the cost function upon optimization completion.
    - cost_evolution (list): Tracks the evolution of the cost function during optimization.
    - temp_cost_evolution (list): Temporarily stores cost values for the current optimization iteration.
    - opt_vec (list): Stores the optimal cost values found in each optimization experiment.
    - x_vec (list): Stores the parameter vectors corresponding to each optimal cost found.
    - number_of_experiments (int): The number of optimization experiments to perform.
    """

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1) -> None:
        """
        Initializes the VQE algorithm instance with a given QUBO matrix and number of layers.
        
        Parameters:
        - qubo_matrix (np.ndarray): The QUBO matrix representing the problem to be solved.
        - layers (int): The number of layers in the variational quantum circuit.
        """
        super().__init__(qubo_matrix, layers)  # Initialize the base class attributes
        self.nc = len(qubo_matrix)  # The number of classical variables is the size of the QUBO matrix
        self.nq = self.nc  # The number of qubits matches the number of classical variables
        self.algorithm = "VQE"  # Set the algorithm name
        # Initialize additional VQE-specific attributes
        self.optimal_params = "-"
        self.cost_evolution = []
        self.temp_cost_evolution = []
        self.opt_vec = []
        self.x_vec = []
        self.number_of_experiments = 1

    def circuit(self, theta: np.ndarray) -> QuantumCircuit:
        """
        Constructs the variational quantum circuit for the VQE algorithm using the given parameters.
        
        Parameters:
        - theta (np.ndarray): The parameters for the variational circuit, including rotation angles.
        
        Returns:
        - QuantumCircuit: The constructed variational quantum circuit.
        """
        qc = QuantumCircuit(self.nq)  # Initialize the quantum circuit with the number of qubits

        for iter in range(self.layers):  # Loop over the layers
            for n in range(self.nq):  # Apply rotation gates to each qubit
                qc.ry(theta[iter * self.nq + n], n)

            # Add entangling gates between qubits if not in the last layer
            if iter < self.layers - 1:
                for n in range(0, self.nq, 2):
                    if n + 1 < self.nq:
                        qc.cx(n, n + 1)
                for n in range(1, self.nq, 2):
                    if n + 1 < self.nq:
                        qc.cx(n, n + 1)
                qc.barrier()  # Add a barrier for visualization

        qc.measure_all()  # Add measurement to all qubits
        return qc

    def compute_expectation(self, counts: dict) -> float:
        """
        Computes the expectation value of the cost function given the measurement outcomes (counts).
        
        Parameters:
        - counts (dict): A dictionary with bitstrings as keys and counts as values.
        
        Returns:
        - float: The computed expectation value of the cost function.
        """
        sum_en = 0
        total_counts = 0
        for bitstr, count in counts.items():
            bitstring = np.array([int(x) for x in bitstr])
            sum_en += count * (bitstring.T @ self.qubo_matrix @ bitstring)  # Calculate energy
            total_counts += count

        return sum_en / total_counts  # Return the average energy

    def get_solution_distribution(self,normalization = [], label = "", marker = '*', solutions=50, shots=10000):
        """
        Samples solutions from the quantum circuit using the optimal parameters and visualizes the distribution 
        of solution qualities in terms of their cost values. It aims to give an insight into the variability 
        and quality of solutions that the VQE algorithm can generate.
        
        Parameters:
        - solutions (int): The number of solutions to sample in each optimization experiment.
        - shots (int): The number of measurement shots for each execution of the quantum circuit.
        
        This method executes the quantum circuit with optimal parameters for a specified number of shots,
        collects the measurement outcomes, calculates the cost for each outcome, and visualizes the distribution 
        of these costs. It helps in understanding how the solutions are distributed across different cost values 
        and how often the optimal or near-optimal solutions are obtained.
        
        Note:
        - This visualization can be particularly useful for analyzing the performance and reliability of the VQE algorithm.
        """
        optimization_runs = []  # Track which optimization run each solution belongs to
        cost_fun = []  # Store the cost function values of sampled solutions
        fraction = []  # Store the fraction of times each solution was observed
        vmin = normalization[0]

        def sample_solutions(params, run, solutions=10):
            """
            Samples a specified number of solutions using the given parameters and updates the lists
            to track the cost values and their occurrences.
            
            Parameters:
            - params (np.ndarray): The parameters with which to execute the quantum circuit.
            - run (int): The current optimization run number.
            - solutions (int): The number of solutions to sample.
            """
            nonlocal cost_fun, fraction, optimization_runs

            temp_cost_fun = []  # Temporary storage for cost function values in this sampling
            temp_fraction = []  # Temporary storage for fractions in this sampling
            temp_optimization_runs = []  # Temporary storage for optimization run tracking

            # Execute the quantum circuit with the given parameters
            vqa_circ = self.circuit(params)
            for i in range(solutions):
                job = execute(vqa_circ, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=shots, max_parallel_threads=1)
                counts = job.result().get_counts(vqa_circ)
                sol = max(counts, key=counts.get)  # Get the most frequent measurement result

                bitstring = np.array([int(x) for x in sol])
                cost = bitstring.T @ self.qubo_matrix @ bitstring -vmin # Calculate the cost for this solution

                if cost in temp_cost_fun:
                    ind = temp_cost_fun.index(cost)
                    temp_fraction[ind] += 1
                else:
                    temp_cost_fun.append(cost)
                    temp_fraction.append(1)
                    temp_optimization_runs.append(run)
                
            # Update the global lists with data from this sampling
            cost_fun += temp_cost_fun
            fraction += temp_fraction
            optimization_runs += temp_optimization_runs

        # Sample solutions for each optimization experiment
        for run in range(1, self.number_of_experiments + 1):
            sample_solutions(self.x_vec[run - 1], run, solutions)

        fractions = [element / solutions for element in fraction]  # Calculate the fraction of each solution

        # Create a scatter plot to visualize the distribution of solution qualities
        # plt.figure(figsize=(10, 6))
        scatter = plt.scatter(cost_fun, optimization_runs, c=fractions, cmap='viridis', marker='^', label=self.algorithm)
        # plt.xlabel('Cost Function Value')
        # plt.ylabel('Optimization Run')
        # plt.title('Distribution of Solution Qualities in VQE Optimization')
        # plt.grid(True)
        # plt.show()

        return scatter


    def show_solution(self):
        """
        Executes the quantum circuit with the optimal parameters to find and display the best solution.
        
        Returns:
        - str: The bitstring representing the best solution found.
        """
        vqa_circ = self.circuit(self.optimal_params)
        job = execute(vqa_circ, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=10000, max_parallel_threads=1)
        count = job.result().get_counts(vqa_circ)
        sol = max(count, key=count.get)
        print(f"QUBO SOLUTION IS: {sol}")
        return sol

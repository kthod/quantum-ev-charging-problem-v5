from qiskit import *
from optimizationAlgorithm import optimizationAlgorithm
import numpy as np
import matplotlib.pyplot as plt

class CompressedVQE(optimizationAlgorithm):
    """
    Implements a compressed version of the Variational Quantum Eigensolver (VQE) algorithm to solve QUBO problems with a reduced number of qubits.
    
    This class extends optimizationAlgorithm and modifies the VQE approach to allow for a compressed encoding of the solution space, aiming to reduce the quantum resource requirements by using fewer qubits than the number of classical variables.
    
    Attributes inherited and new attributes specific to CompressedVQE:
    - nc (int): Number of classical variables in the QUBO problem.
    - nr (int): Number of required register qubits for the compressed encoding.
    - na (int): Number of ancilla qubits used for the compression.
    - nq (int): Total number of qubits used in the quantum circuit (nr + na).
    - optimal_params (str or np.ndarray): Parameters that minimize the cost function.
    - cost_evolution, temp_cost_evolution, opt_vec, x_vec: Lists for tracking the optimization process.
    - number_of_experiments (int): Specifies how many times the optimization should be run.
    
    The class constructor checks if the compression is feasible with the given problem size and setup parameters.
    """

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1, na: int = 1) -> None:
        """
        Initializes the CompressedVQE instance with a QUBO problem matrix, number of layers in the variational circuit, and the number of ancilla qubits for compression.
        
        Parameters:
        - qubo_matrix (np.ndarray): The QUBO matrix representing the optimization problem.
        - layers (int): The number of layers in the variational quantum circuit.
        - na (int): The number of ancilla qubits to be used for compression.
        
        Raises:
        - ValueError: If the number of ancilla qubits equals the size of the QUBO matrix, suggesting full encoding should be used instead.
        """
        super().__init__(qubo_matrix, layers)
        if na == len(qubo_matrix):
            raise ValueError("For full encoding use VQE instead.")

        self.nc = len(qubo_matrix)
        print(self.nc)
        self.nr = int(np.ceil(np.log2(self.nc / na)))
        self.na = na
        self.nq = self.nr + na
        self.algorithm = "Compressed VQE"
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

    def get_solution_distribution(self,normalization =[], label = "", marker = '*', solutions=10, shots=10000):
        """
        Samples solutions from the quantum circuit and visualizes the distribution of solution qualities 
        in terms of their cost values. This method helps in understanding the variability and quality of 
        solutions that the Compressed VQE algorithm can generate.

        Parameters:
        - solutions (int, optional): The number of solutions to sample in each optimization experiment. 
                                    Defaults to 10.
        - shots (int, optional): The number of measurement shots for each execution of the quantum circuit. 
                                Defaults to 10000.

        This method executes the quantum circuit multiple times with the optimal parameters to sample 
        different solutions. For each sampled solution, it calculates the corresponding cost using the 
        QUBO matrix. Then, it visualizes the distribution of these costs across different optimization 
        experiments to provide insights into the algorithm's performance.

        The visualization is created as a scatter plot, where each point represents a sampled solution, 
        with its cost on one axis and the optimization run it belongs to on another. The color of each point 
        reflects the frequency of each cost value being observed, providing a visual representation of the 
        solution quality distribution.

        Returns:
        - matplotlib.collections.PathCollection: The scatter plot object showing the distribution of solution 
                                                qualities across optimization runs.
        
        Note:
        - This method assumes that the optimization process has been completed, and optimal parameters for 
        the quantum circuit are available in `self.x_vec`.
        - The visualization can be particularly useful for analyzing the effectiveness and reliability of the 
        Compressed VQE algorithm in finding good solutions to the QUBO problem.
        """
        optimization_runs = []  # Tracks which optimization run each solution belongs to
        cost_fun = []  # Stores the cost function values of sampled solutions
        fraction = []  # Stores the fraction of times each solution was observed
        vmin = normalization[0]

        if label == "":
            label = self.algorithm
        # Define an inner function to sample solutions and calculate their costs
        def sample_solutions(params, run, solutions=50):
            nonlocal cost_fun, fraction, optimization_runs
            temp_cost_fun = []  # Temporary storage for this sampling
            temp_fraction = []  # Temporary storage for fractions in this sampling
            temp_optimization_runs = []  # Temporary storage for optimization run tracking
            
            # Execute the circuit with given parameters and sample solutions
            vqa_circ = self.circuit(params)
            for _ in range(solutions):
                job = execute(vqa_circ, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=shots, max_parallel_threads=1)
                count = job.result().get_counts(vqa_circ)
                probabilities = self.calculate_probabilities(count)
                
                # Reconstruct the solution bitstring from the most probable ancilla configurations
                sol = ""
                for i in range(len(probabilities)):
                    ind = np.argmax(probabilities[i,:])
                    ancilla = f'{ind:0{self.na}b}'
                    sol += ancilla
                
                bitstring = np.array([int(x) for x in sol])
                cost = bitstring.T @ self.qubo_matrix @ bitstring -vmin
                
                # Update the distributions
                if cost in temp_cost_fun:
                    ind = temp_cost_fun.index(cost)
                    temp_fraction[ind] += 1
                else:
                    temp_cost_fun.append(cost )
                    temp_fraction.append(1)
                    temp_optimization_runs.append(run)
                
            cost_fun.extend(temp_cost_fun)
            fraction.extend(temp_fraction)
            optimization_runs.extend(temp_optimization_runs)

        # Sample solutions across all optimization experiments
        for run in range(1, self.number_of_experiments + 1):
            sample_solutions(self.x_vec[run - 1], run, solutions)
        
        fractions = [element / solutions for element in fraction]  # Normalize fractions

        # Visualize the solution distribution
        scatter = plt.scatter(cost_fun, optimization_runs, c=fractions, cmap='viridis', marker=marker, label=label)
        # Additional plot formatting can be added here

        return scatter



    def show_solution(self):
        """
        Executes the quantum circuit with the optimal parameters and interprets the resulting quantum state to 
        find and display the optimal solution for the QUBO problem.

        This method performs the following steps:
        1. Constructs and executes the variational quantum circuit using the optimal parameters found during 
        the optimization process.
        2. Processes the measurement outcomes (counts) to calculate the probabilities of different ancilla 
        configurations.
        3. Determines the most probable configuration for each set of ancilla qubits, reconstructing the 
        solution bitstring for the original QUBO problem.
        4. Displays the constructed solution bitstring as the output.

        The method leverages the `calculate_probabilities` method to translate quantum measurement outcomes 
        into probabilities for each possible state of the ancilla qubits, which are then used to find the 
        most probable solution state.

        Returns:
        - str: The bitstring representing the optimal solution found by the algorithm. This solution is 
            also printed to the console for direct observation.

        Note:
        - The solution is represented as a bitstring where each bit corresponds to a decision variable 
        in the QUBO problem. The length and content of the bitstring depend on the specific problem 
        being solved and the encoding used in the quantum circuit.
        - This method assumes that the optimal parameters (`self.optimal_params`) have already been 
        determined through the optimization process.
        """
        # Construct the quantum circuit with the optimal parameters
        vqa_circ = self.circuit(self.optimal_params)
        
        # Execute the circuit on a quantum simulator with a high number of shots for accuracy
        job = execute(vqa_circ, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=100000, max_parallel_threads=1)
        count = job.result().get_counts(vqa_circ)
        
        # Calculate probabilities for each possible ancilla configuration
        probabilities = self.calculate_probabilities(count)
        
        # Construct the solution bitstring based on the most probable configurations
        sol = ""
        for i in range(len(probabilities)):
            ind = np.argmax(probabilities[i, :])  # Find the index of the max probability
            ancilla = f'{ind:0{self.na}b}'  # Convert the index to a binary string
            sol += ancilla  # Append this ancilla state to the solution bitstring
        
        # Display the solution
        print(f"QUBO SOLUTION IS: {sol}")
        return sol
    
    def calculate_probabilities(self, counts):
        """
        Calculates the probabilities of each possible ancilla configuration based on the measurement outcomes from the quantum circuit.

        Given the counts (measurement outcomes) from executing the quantum circuit, this method computes the probability distribution
        for the ancilla qubits' states. These probabilities are essential for interpreting the quantum solution in terms of the
        classical problem variables.

        Parameters:
        - counts (dict): A dictionary where keys are bitstrings representing the outcomes of the quantum circuit execution, and values
                        are the number of times each bitstring was observed.

        Returns:
        - np.ndarray: A 2D array of probabilities, where each row corresponds to a register and each column to an ancilla configuration.
                    The element at position (i, j) represents the probability of the j-th ancilla configuration in the i-th register.

        The method divides the classical problem variables into registers based on the ancilla length (na) and calculates the probability
        of encountering each ancilla state within these registers. It ensures that even unobserved states (those with zero counts) are
        accounted for by assigning them a minimal probability, thus avoiding division by zero and ensuring a complete probabilistic
        description of the solution space.

        Note:
        - This method assumes a compressed encoding scheme where the solution space is partitioned into registers, each represented by
        a subset of ancilla qubits. The method is specific to the 'Compressed VQE' approach, where the goal is to solve large QUBO
        problems with a reduced number of qubits.
        """
        ancilla_len = self.na  # The number of ancilla qubits used for encoding
        nc = self.nc  # The number of classical variables in the QUBO problem

        # Calculate the number of registers needed to represent the classical variables
        num_registers = int(nc / ancilla_len)
        # Initialize a probability matrix with zeros
        probabilities = np.zeros((num_registers, 2**ancilla_len))
        # Counter for the number of observations per register
        counter = np.zeros(num_registers)

        for bitstr, count in counts.items():
            # Extract the ancilla state and the register index from the bitstring
            ancilla = int(bitstr[0:ancilla_len], 2)
            register = int(bitstr[ancilla_len:], 2)

            # Skip counts that do not correspond to a valid register
            if register >= num_registers:
                continue
            # Update the observation counter and probabilities based on the measurement outcomes
            counter[register] += count
            probabilities[register][ancilla] += count
        
        # Adjust for any registers that were not observed
        if np.any(counter == 0):
            ind = np.where(counter == 0)[0]
            counter[ind] = 2  # Assign a minimal count to avoid division by zero
            probabilities[ind, :] = 1  # Assign equal probability to all states in unobserved registers

        # Normalize the probabilities by the total counts for each register
        for i in range(len(counter)):
            probabilities[i, :] /= counter[i]

        return probabilities
    
    def compute_expectation(self, counts: dict) -> float:
        """
        Computes the expected value of the QUBO problem's cost function based on the outcomes of quantum circuit measurements.

        This method processes the measurement outcomes (counts) to calculate the expectation value of the cost function
        for the given QUBO matrix. It takes into account the compressed encoding of the problem variables and the specific
        structure of the quantum circuit to accurately estimate the expected cost.

        Parameters:
        - counts (dict): A dictionary where keys are bitstrings representing outcomes from the quantum circuit execution,
                        and values are the counts of each bitstring observed.

        Returns:
        - float: The calculated expectation value of the cost function.

        The computation involves:
        - Parsing the bitstring outcomes to extract ancilla and register information.
        - Accumulating counts to calculate probabilities of ancilla states contributing to the cost.
        - Applying these probabilities to the QUBO matrix to compute the expectation value.
        - Adjusting for the encoding scheme to ensure correct calculation across different register segments.

        Note:
        - This method is specifically designed for the Compressed VQE approach, where a portion of the solution
        space is encoded into a smaller number of qubits (ancilla qubits) to reduce the quantum resource requirements.
        - It assumes that the ancilla and register qubits' roles are predefined and that the QUBO problem is suitably
        encoded into the quantum circuit.
        """
        na = self.na  # Number of ancilla qubits used for compression
        nc = len(self.qubo_matrix)  # Total number of classical variables in the QUBO problem

        # Calculate the number of logical registers based on compression
        num_registers = int(np.ceil(nc / na))

        # Initialize matrices to track counts and probabilities for pairs of qubits
        register_counts = np.zeros((nc, nc), dtype=int)
        P = np.zeros((nc, nc), dtype=float)

        # Process each bitstring and its associated count
        for bitstring, count in counts.items():
            aux = bitstring[:na]  # Extract ancilla part of the bitstring
            reg = int(bitstring[na:], 2)  # Extract register index from the bitstring

            if reg >= num_registers:
                continue  # Skip invalid register indices

            # Accumulate probabilities for each ancilla state within its register
            for i in range(na):
                for j in range(i, na):
                    if aux[i] == '1' and aux[j] == '1':
                        P[reg * na + i][reg * na + j] += count
                    register_counts[reg * na + i][reg * na + j] += count

        # Avoid division by zero by setting unobserved combinations to a minimal count
        register_counts[np.where(register_counts == 0)] = 1
        P /= register_counts  # Normalize to get probabilities

        # Adjust probabilities for qubits in different registers based on observed data
        for i in range(nc - 1):
            for j in range(i + 1, nc):
                reg_index_i = i // na
                reg_index_j = j // na
                if reg_index_i != reg_index_j:
                    P[i][j] = P[i][i] * P[j][j]  # Assume independence between different registers

        # Calculate the expected value by applying the probabilities to the QUBO matrix
        return np.sum(P * self.qubo_matrix)



from optimizationAlgorithm import *
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

class CompressedVQE (optimizationAlgorithm):

    def __init__(self, qubo_matrix: np.array, layers: int = 1, na: int = 1) -> None:
         
        super().__init__(qubo_matrix, layers)

        if na == len(qubo_matrix):
            raise ValueError("For full encoding use VQE instead.")
        

        self.nc = len(qubo_matrix)
        self.nr = int(np.ceil(np.log2(self.nc/na)))
        self.na = na

        self.nq = self.nr + na
        self.algorithm = "Compressed VQE"
        self.optimal_params = "-"
        self.cost_evolution = []
        self.temp_cost_evolution = []

        self.opt_vec =[]
        self.x_vec = []
        self.number_of_experiments = 1

    
    def circuit(self, theta): # Creating our circuit
        qc = QuantumCircuit(self.nq)

        for iter in range(self.layers):
            for n in range(self.nq):
                qc.ry(theta[iter * self.nq + n], n)

            if iter < self.layers - 1:
                for n in range(0, self.nq, 2):
                    if n+1 < self.nq:
                        qc.cx(n, n+1)

                for n in range(1, self.nq, 2):
                    if n+1 < self.nq:
                        qc.cx(n, n+1)
                
                qc.barrier()
        
        qc.measure_all()
        return qc

    def get_solution_distribution(self,solutions = 10,shots =10000):
        
        optimization_runs = []
        cost_fun = []
        fraction = []

        def sample_solutions(params, run, solutions=50):

            nonlocal cost_fun, fraction, optimization_runs

            temp_cost_fun = []
            temp_fraction = []
            temp_optimization_runs = []
            vqa_circ = self.circuit(params)
            for i in range(solutions):
                job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=shots,  max_parallel_threads=1)
                count = job.result().get_counts(vqa_circ)
                ancilla_len = self.na
                probabilities = self.calculate_probabilities(count)
                sol = ""
                for i in range(len(probabilities)):
                    ind = np.argmax(probabilities[i,:]) 
                    ancilla = f'{ind:0{ancilla_len}b}'
                    sol+=ancilla

                bitstring = np.array([int(x) for x in sol])

                cost = bitstring.T @ self.qubo_matrix @ bitstring

                if cost in  temp_cost_fun:
                    ind =  temp_cost_fun.index(cost)
                    temp_fraction[ind]+=1
                else :
                    temp_cost_fun.append(cost)
                    temp_fraction.append(1)
                    temp_optimization_runs.append(run)
                
                cost_fun+=temp_cost_fun
                fraction+=temp_fraction
                optimization_runs+=temp_optimization_runs
    

        # Your data here
                
        for run in range(1,self.number_of_experiments+1):
            sample_solutions(self.x_vec[run-1],run,solutions)
     
        fractions = [element /solutions for element in fraction]

        # Create a scatter plot
        # plt.figure(figsize=(8, 6))
        scatter = plt.scatter(cost_fun, optimization_runs, c=fractions, cmap='viridis', marker='*', label=self.algorithm)
        #plt.scatter(c_norm, optimization_runs, c=fractions, cmap='viridis', marker='*', label='minimal encoding')

        # Creating a colorbar
        # plt.colorbar(scatter)
        # plt.yticks(optimization_runs)
        # # Labels and title
        # plt.xlabel('C_norm')
        # plt.ylabel('Optimization run')
        # plt.title('Distribution of solutions for nc = 16 routes')
        # plt.grid(True)
        # plt.show()
        return scatter

    def show_solution(self):
        vqa_circ = self.circuit(self.optimal_params)
        job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=100000,  max_parallel_threads=1)
        count = job.result().get_counts(vqa_circ)
        ancilla_len = self.na
        probabilities = self.calculate_probabilities(count)
        sol = ""
        for i in range(len(probabilities)):
            ind = np.argmax(probabilities[i,:]) 
            ancilla = f'{ind:0{ancilla_len}b}'
            sol+=ancilla

        print(f"QUBO SOLUTION IS :{sol}")
        return sol
    # Additional functions to calculate the cost function value so we can pass it
    # to our optimizer (COBYLA).

    
    def calculate_probabilities(self,counts):

        ancilla_len = self.na
        nc = self.nc

        num_registers = int(nc/ancilla_len)
        probabilities = np.zeros((num_registers, 2**ancilla_len))

        
        # counter_01s = np.zeros(int(nc/2))
        # counter_10s = np.zeros(int(nc/2))
        # counter_11s = np.zeros(int(nc/2))
        counter = np.zeros(num_registers)
        for bitstr, count in counts.items():
            
            ancilla = int(bitstr[0:ancilla_len],2)
            
            register = int(bitstr[ancilla_len:],2)

            if register >= num_registers:
                    continue
            counter[register]+=count
            probabilities[register][ancilla]+=count
            
        if np.any(counter==0):
            ind = np.where(counter == 0)[0]

            counter[ind] = 2
            probabilities[ind,:] = 1

        for i in range(len(counter)):
            probabilities[i, :] /= counter[i]
            # print(probabilities)
            # ow_sums = np.sum(probabilities, axis=1)
            # print(ow_sums)
        return probabilities
    def compute_expectation(self,counts: dict) -> float:

            na = self.na
            nc = len(self.qubo_matrix)

            num_registers = int(np.ceil(nc/na))

            register_counts = np.zeros((nc, nc), dtype=int)
            P = np.zeros((nc, nc), dtype=float)

            for bitstring, count in counts.items():

                aux = bitstring[:na]
                reg = int(bitstring[na:], 2)

                if reg >= num_registers:
                    continue

                for i in range(na):
                    for j in range(i, na):
                        if aux[i] == '1' and aux[j] == '1':
                            P[reg * na + i][reg * na + j] += count
                        register_counts[reg * na + i][reg * na + j] += count

            register_counts[np.where(register_counts == 0)] = 1

            P /= register_counts

            for i in range(nc-1):
                for j in range(i+1, nc):
                    reg_index_i = i // na
                    reg_index_j = j // na

                    if reg_index_i != reg_index_j:
                        P[i][j] = P[i][i] * P[j][j]

            return np.sum(P * self.qubo_matrix)
    

    
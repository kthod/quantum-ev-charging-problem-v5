from qiskit import *
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize
from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from abc import ABC, abstractclassmethod

class OptimizationAlgorithm(ABC):

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1) -> None:
        self.qubo_matrix = qubo_matrix
        self.layers = layers
        self.nq = 0

        self.algorithm = "-"
        self.status = "INITIALIZED"
        self.optimal_answer = "-"
        self.optimal_cost = 0

        self.cost_evolution = list()

    def __str__(self) -> str:
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
    def build_circuit(self, theta: list) -> QuantumCircuit:
        raise NotImplementedError()
    
    @abstractclassmethod
    def optimize(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
        raise NotImplementedError()
    
    def plot_evolution(self, normalization = []) -> None:
        assert len(normalization) == 0 or len(normalization) == 2

        measurements = np.array(self.cost_evolution)
        maxiter = len(measurements[0])

        if len(normalization) == 2:
            vmin = normalization[0]
            vmax = normalization[1]

            measurements = (measurements - np.array([[vmin] * maxiter] * len(measurements))) / np.array([[vmax - vmin] * maxiter] * len(measurements))

        upper_bound = [np.max(measurements[:, i]) for i in range(maxiter)]
        lower_bound = [np.min(measurements[:, i]) for i in range(maxiter)]
        mean = [np.mean(measurements[:, i]) for i in range(maxiter)]
        
        plt.figure()
        plt.fill_between(range(maxiter), upper_bound, lower_bound, color='lightblue')
        plt.plot(range(maxiter), mean, color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
   
    def phow_solution(self):
            qc = self.build_circuit(self.optimal_answer)
            backend = Aer.get_backend('qasm_simulator')
            counts = execute(qc, backend).result().get_counts()
            probabilities = self.calculate_probabilities(counts)
            sol = ""
            for i in range(len(probabilities)):
                ind = np.argmax(probabilities[i,:]) 
                ancilla = f'{ind:0{self.na}b}'
                sol+=ancilla

            print(f"QUBO SOLUTION IS :{sol}")
            plot_histogram(counts, figsize=(7,5))
            plt.show()

class CompressedVQE(OptimizationAlgorithm):

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1, na: int = 1) -> None:
        
        if na == len(qubo_matrix):
            raise ValueError("For full encoding use VQE instead.")
        
        super().__init__(qubo_matrix, layers)

        self.nc = len(qubo_matrix)
        nr = int(np.ceil(np.log2(self.nc/na)))
        self.na = na

        self.nq = nr + na
        self.algorithm = "Compressed VQE"

        self.temp_cost_evolution = None

    def __str__(self) -> str:
        return super().__str__()
    
    def build_circuit(self, theta: list) -> QuantumCircuit:
        
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

    def calculate_probabilities(self,counts):
            nc = self.nc
            na =self.na
            probabilities = np.zeros((int(nc/na), 2**na))
            # counter_01s = np.zeros(int(nc/2))
            # counter_10s = np.zeros(int(nc/2))
            # counter_11s = np.zeros(int(nc/2))
            num_registers = int(np.ceil(nc/na))
            counter = np.zeros(int(nc/na))
            for bitstr, count in counts.items():
                
                ancilla = int(bitstr[0:na],2)
                
                register = int(bitstr[na:],2)
                
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

    def optimize(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:

        
        def cost_function(counts): # calculates cost function for max cut            nc = self.nc
   #num_of_registers = int(len(a)/q)
            nc = self.nc
            ancilla_len = self.na
            P =  np.zeros((nc,nc))
            probabilities = self.calculate_probabilities(counts)
            for i in range(nc):
                
                for j in range(i,nc):
                    
                    anc_index_i = i%ancilla_len
                    reg_index_i = i//ancilla_len
                    anc_index_j = j%ancilla_len
                    reg_index_j = j//ancilla_len

                    
                    if reg_index_i == reg_index_j:
                        #print("start")
                        for ancilla_int in range(2**ancilla_len):
                            ancilla = f'{ancilla_int:0{ancilla_len}b}'
                            #print(ancilla)
                            if ancilla[anc_index_i]=='1' and ancilla[anc_index_j]=='1':
                                #print(f"i {i} j {j}")
                                P[i][j] += probabilities[reg_index_i][ancilla_int]
                        
                    else:
                        p_i=0
                        for ancilla_int in range(2**ancilla_len):
                            ancilla = f'{ancilla_int:0{ancilla_len}b}'
                            if ancilla[anc_index_i]=='1':
                                p_i += probabilities[reg_index_i][ancilla_int]

                        p_j=0        
                        for ancilla_int in range(2**ancilla_len):
                            ancilla = f'{ancilla_int:0{ancilla_len}b}'
                            if ancilla[anc_index_j]=='1':
                                p_j += probabilities[reg_index_j][ancilla_int]

                        P[i][j]=p_i*p_j

                

            # x = np.array(x)
            # a= np.array(a)
            # print(a)
            # print(x)
            Q = self.qubo_matrix
            obj =0
            for i in range(self.nc):
                for j in range(nc):
                    
                    obj+=Q[i][j]*P[i][j]
                
            
       
    #print(obj)      
            return obj

        def objective():
            backend = Aer.get_backend('qasm_simulator')

            def execute_circ(theta):
                
                qc = self.build_circuit(theta)
                counts = execute(qc, backend, shots=shots).result().get_counts()
                exp = cost_function(counts)
                self.temp_cost_evolution.append(-exp)
                return -exp
            
            return execute_circ
        opt_vec =[]
        x_vec = []       
        for _ in range(experiments):

            self.temp_cost_evolution = list()

            obj = objective()
            theta = np.random.uniform(low=0, high=np.pi, size=self.layers*self.nq)
            result = minimize(obj, theta, method="COBYLA", options={"maxiter": maxiter})

            self.cost_evolution.append(np.asarray((self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution)))))

            print('Experiment:', (_+1), "min:", result.fun)
            opt_vec.append(result.fun)
            x_vec.append(result.x)
            # Execute the final quantum circuit one more time using the optimized parameters.
            

        self.optimal_cost = np.min(opt_vec)
        ind = np.argmin(opt_vec)
        self.optimal_answer = x_vec[ind]


                # def compute_expectation(counts: dict) -> float:

        #     na = self.na
        #     nc = len(self.qubo_matrix)

        #     num_registers = int(np.ceil(nc/na))

        #     register_counts = np.zeros((nc, nc), dtype=int)
        #     P = np.zeros((nc, nc), dtype=float)

        #     for bitstring, count in counts.items():

        #         aux = bitstring[:na]
        #         reg = int(bitstring[na:], 2)

        #         if reg >= num_registers:
        #             continue

        #         for i in range(na):
        #             for j in range(i, na):
        #                 if aux[i] == '1' and aux[j] == '1':
        #                     P[reg * na + i][reg * na + j] += count
        #                 register_counts[reg * na + i][reg * na + j] += count

        #     register_counts[np.where(register_counts == 0)] = 1

        #     P /= register_counts

        #     for i in range(nc-1):
        #         for j in range(i+1, nc):
        #             reg_index_i = i // na
        #             reg_index_j = j // na

        #             if reg_index_i != reg_index_j:
        #                 P[i][j] = P[i][i] * P[j][j]

        #     return np.sum(P * self.qubo_matrix)
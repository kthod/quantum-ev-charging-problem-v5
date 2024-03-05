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

from abc import ABC, abstractclassmethod

class optimizationAlgorithm(ABC):

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
    def circuit(self, theta: list) -> QuantumCircuit:
        raise NotImplementedError()
    
    @abstractclassmethod
    def cost_function(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
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
        
        #plt.figure()
        plt.fill_between(range(maxiter), upper_bound, lower_bound, alpha=0.5)
        plt.plot(range(maxiter), mean, linestyle = '--', label = self.algorithm)
        # plt.xlabel('Iteration')
        # plt.ylabel('Cost')
        # plt.show()

    def cost_function(self, parameters, number_of_measurements):
        
        
        # Number of qubits = number of variables = length of matrix
        circ = self.circuit(parameters)
        #vqa_circuit.draw(output='mpl')
        job = execute(circ,Aer.get_backend('qasm_simulator'),shots=number_of_measurements,  max_parallel_threads=1)
        counts = job.result().get_counts(circ)
        
 
        # probabilities = self.calculate_probabilities(counts)
        # obj = self.subset_sum_obj(probabilities)
            
        obj = self.compute_expectation(counts)
        self.temp_cost_evolution.append(obj)
        return obj


    def optimize(self, n_measurements = 10000, number_of_experiments = 10, maxiter = 150):


        self.number_of_experiments = number_of_experiments
        sum_time=0
        for k in range(number_of_experiments):
            self.temp_cost_evolution=[]
            initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.nq)

            print(f"Begin optimization for hardware efficient {self.algorithm} and {n_measurements} measurments...")
            time1 = time.time()
            opt = minimize(self.cost_function,initial_vector, args=(n_measurements), method = 'COBYLA',options={'maxiter': maxiter})
            sum_time += time.time() - time1
            print("Optimization complete. Time taken: %.3fs." %(time.time()-time1))
            print()
            self.opt_vec.append(opt.fun)
            self.x_vec.append(opt.x)
            #self.sample_solutions(opt.x, k+1)
            print("Final cost function value: %.5f" %opt.fun)
            print("Number of optimizer iterations: %.d" %opt.nfev)
            #print("Optimal parameters:", opt.x%(2*np.pi))
            
            self.cost_evolution.append(np.asarray((self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution)))))

        # print(sum_time/number_of_experiments)    
        # print(np.min(opt_vec))
        # print(opt_vec)
        ind = np.argmin(self.opt_vec)
        self.optima = self.x_vec[ind]
        # maxcf_mc = [max(values) for values in zip(*self.cost_evolution)]
        # mincf_mc = [min(values) for values in zip(*self.cost_evolution)]
        # meancf_mc = [sum(values)/number_of_experiments for values in zip(*self.cost_evolution)]
        #plt.plot(maxcf)
        #plt.plot(mincf)
        self.optimal_params = self.x_vec[ind]
        # plt.plot(meancf_mc,linestyle='--',label = 'Compressed VQE mean')

        # plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        # plt.show()

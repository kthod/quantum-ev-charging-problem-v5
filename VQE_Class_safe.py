from qiskit import *
from optimizationAlgorithm import *
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize
from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import time

class VQE(optimizationAlgorithm):

    def __init__(self, qubo_matrix: np.array, layers: int = 1) -> None:
        
        self.qubo_matrix = qubo_matrix
        self.layers = layers

        

        self.nc = len(qubo_matrix)
   

        self.nq = self.nc
        self.algorithm = "VQE"
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

    def get_solution_distribution(self,solutions = 50,shots = 10000):
        # Your data here
        optimization_runs = []
        cost_fun = []
        fraction = []

        def sample_solutions(params, run, solutions=10):

            nonlocal cost_fun, fraction, optimization_runs

            temp_cost_fun = []
            temp_fraction = []
            temp_optimization_runs = []
            vqa_circ = self.circuit(params)
            for i in range(solutions):
                job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=shots,  max_parallel_threads=1)
                count = job.result().get_counts(vqa_circ)
                sol = max(count, key=count.get)

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

        for run in range(1,self.number_of_experiments+1):
            sample_solutions(self.x_vec[run-1],run,solutions)

        fractions = [element /solutions for element in fraction]

        # Create a scatter plot
        #plt.figure(figsize=(8, 6))
        scatter = plt.scatter(cost_fun, optimization_runs, c=fractions, cmap='viridis', marker='^', label=self.algorithm)
        #plt.scatter(c_norm, optimization_runs, c=fractions, cmap='viridis', marker='*', label='minimal encoding')

        # Creating a colorbar


        return scatter
        # Labels and title
       


    def show_solution(self):
        vqa_circ = self.circuit(self.optimal_params)
        job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=10000,  max_parallel_threads=1)
        count = job.result().get_counts(vqa_circ)
        sol = max(count, key=count.get)

        print(f"QUBO SOLUTION IS :{sol}")
        return sol
    # Additional functions to calculate the cost function value so we can pass it
    # to our optimizer (COBYLA).

    

    
    def compute_expectation(self,counts: dict) -> float:


            sum_en = 0
            total_counts = 0
            for bitstr, count in counts.items():

                bitstring = np.array([int(x) for x in bitstr])

                sum_en += count*(bitstring.T @ self.qubo_matrix @ bitstring)
                total_counts += count


            return sum_en/total_counts
    

    # def cost_function(self, parameters, number_of_measurements):
        
        
    #     # Number of qubits = number of variables = length of matrix
    #     vqa_circuit = self.circuit(parameters)
    #     #vqa_circuit.draw(output='mpl')
    #     job = execute(vqa_circuit,Aer.get_backend('qasm_simulator'),shots=number_of_measurements,  max_parallel_threads=1)
    #     counts = job.result().get_counts(vqa_circuit)
        
        
    #     # probabilities = self.calculate_probabilities(counts)
    #     # obj = self.subset_sum_obj(probabilities)
            
    #     obj = self.compute_expectation(counts)
    #     self.temp_cost_evolution.append(obj)
    #     return obj


    # def optimize(self, n_measurements = 1000, number_of_experiments = 5, maxiter = 400):

    #     self.number_of_experiments = number_of_experiments

    #     sum_time=0
    #     for k in range(number_of_experiments):
    #         self.temp_cost_evolution=[]
    #         initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.nq)

    #         print("Begin optimization for hardware efficient VQE and %.d measurments..." %(n_measurements))
    #         time1 = time.time()
    #         opt = minimize(self.cost_function,initial_vector, args=(n_measurements), method = 'COBYLA',options={'maxiter': maxiter})
    #         sum_time += time.time() - time1
    #         print("Optimization complete. Time taken: %.3fs." %(time.time()-time1))
    #         print()
    #         self.opt_vec.append(opt.fun)
    #         self.x_vec.append(opt.x)
    #         #self.sample_solutions(opt.x,k+1)
    #         print("Final cost function value: %.5f" %opt.fun)
    #         print("Number of optimizer iterations: %.d" %opt.nfev)
    #         #print("Optimal parameters:", opt.x%(2*np.pi))
            
    #         self.cost_evolution.append(np.asarray((self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution)))))

    #     # print(sum_time/number_of_experiments)    
    #     # print(np.min(opt_vec))
    #     # print(opt_vec)
    #     ind = np.argmin(self.opt_vec)
    #     self.optima = self.x_vec[ind]
    #     # maxcf_mc = [max(values) for values in zip(*self.cost_evolution)]
    #     # mincf_mc = [min(values) for values in zip(*self.cost_evolution)]
    #     # meancf_mc = [sum(values)/number_of_experiments for values in zip(*self.cost_evolution)]
    #     #plt.plot(maxcf)
    #     #plt.plot(mincf)
    #     self.optimal_params = self.x_vec[ind]
    #     # plt.plot(meancf_mc,linestyle='--',label = 'VQE mean')

    #     # plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
    #     #plt.plot(cflistVQA)
    #     #plt.ylim(-4,0)
    #     # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
    #     # plt.ylabel("Cost function value")
    #     # plt.xlabel("Number of COBYLA iterations")

    #     # plt.show()

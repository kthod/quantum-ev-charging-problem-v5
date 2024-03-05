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

class splitVQE:

    def __init__(self, qubo_matrix: list, layers: int = 1, group: int =1) -> None:
        
        self.qubo_matrix = qubo_matrix
        #print(self.qubo_matrix)
        #print(self.qubo_matrix[1])
       
        self.layers = layers

        self.group = group

        self.nc = int(len(qubo_matrix)/group)
   

        self.nq = self.nc
        self.algorithm = "VQE"
        self.optimal_params = "-"
        self.cost_evolution = []
        self.temp_cost_evolution = []


    

    
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



    def show_solution(self):
        sol_list = ""
        for i in range(self.group):
            vqa_circ = self.circuit(self.optimal_params[i*self.nq*self.layers:(i+1)*self.nq*self.layers])
            job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=10000,  max_parallel_threads=1)
            count = job.result().get_counts(vqa_circ)
            sol = max(count, key=count.get)
            sol_list+=sol
            print(f"QUBO SOLUTION IS :{sol}")
        
        return sol_list

    # Additional functions to calculate the cost function value so we can pass it
    # to our optimizer (COBYLA).


    
    def compute_expectation(self,i:int,counts: dict) -> float:


            sum_en = 0
            total_counts = 0
            for bitstr, count in counts.items():

                bitstring = np.array([int(x) for x in bitstr])

                sum_en += count*(bitstring.T @ self.qubo_matrix[i*self.nq:(i+1)*self.nq,i*self.nq:(i+1)*self.nq] @ bitstring)
                total_counts += count


            return sum_en/total_counts
    def compute_expectation2(self,i:int,j: int, counts_i: dict, counts_j: dict) -> float:


            sum_en = 0
            total_counts = 0
            for bitstri, counti in counts_i.items():
                bitstringi = np.array([int(x) for x in bitstri])
                for bitstrj,countj in counts_j.items():
                    bitstringj = np.array([int(x) for x in bitstrj])

                    sum_en += counti*countj*(bitstringi.T @ self.qubo_matrix[i*self.nq:(i+1)*self.nq,j*self.nq:(j+1)*self.nq] @ bitstringj)
                    total_counts += counti*countj


            return sum_en/total_counts
    

    def vqa_cf(self, parameters, number_of_measurements):
        
        
        # Number of qubits = number of variables = length of matrix
        obj = 0
        count_list = []
        for i in range(self.group):
            vqa_circuit = self.circuit(parameters[i*self.nq*self.layers:(i+1)*self.nq*self.layers])
        #vqa_circuit.draw(output='mpl')
            job = execute(vqa_circuit,Aer.get_backend('qasm_simulator'),shots=number_of_measurements,  max_parallel_threads=1)
            counts = job.result().get_counts(vqa_circuit)
            count_list.append(counts)
        
        # probabilities = self.calculate_probabilities(counts)
        # obj = self.subset_sum_obj(probabilities)
        for i in range(len(count_list)):
            for j in range(i,len(count_list)):
                if i == j :
                    obj += self.compute_expectation(i,count_list[i])
                else:
                    obj +=self.compute_expectation2(i,j,count_list[i],count_list[j])

        self.temp_cost_evolution.append(obj)
        return obj


    def optimize(self, n_measurements = 10000, number_of_experiments = 1, maxiter = 1000):

        opt_vec =[]
        x_vec = []

        sum_time=0
        for k in range(number_of_experiments):
            self.temp_cost_evolution=[]
            initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.nq*self.group)

            print("Begin optimization for hardware efficient ClusterVQE and %.d measurments..." %(n_measurements))
            time1 = time.time()
            opt = minimize(self.vqa_cf,initial_vector, args=(n_measurements), method = 'COBYLA',options={'maxiter': maxiter})
            sum_time += time.time() - time1
            print("Optimization complete. Time taken: %.3fs." %(time.time()-time1))
            print()
            opt_vec.append(opt.fun)
            x_vec.append(opt.x)
            print("Final cost function value: %.5f" %opt.fun)
            print("Number of optimizer iterations: %.d" %opt.nfev)
            #print("Optimal parameters:", opt.x%(2*np.pi))
            
            self.cost_evolution.append(np.asarray((self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution)))))

        print(sum_time/number_of_experiments)    
        print(np.min(opt_vec))
        print(opt_vec)
        ind = np.argmin(opt_vec)
        self.optima = x_vec[ind]
        maxcf_mc = [max(values) for values in zip(*self.cost_evolution)]
        mincf_mc = [min(values) for values in zip(*self.cost_evolution)]
        meancf_mc = [sum(values)/number_of_experiments for values in zip(*self.cost_evolution)]
        #plt.plot(maxcf)
        #plt.plot(mincf)
        self.optimal_params = x_vec[ind]
        plt.plot(meancf_mc,linestyle='--',label = 'ClusterVQE mean')

        plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        # plt.show()


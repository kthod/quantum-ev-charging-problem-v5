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

class CompressedVQE_RPA:

    def __init__(self, qubo_matrix: np.array, layers: int = 1, na: int = 1) -> None:
        
        self.qubo_matrix = qubo_matrix
        self.layers = layers

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


        self.optimization_runs = []
        self.cost_fun = []
        self.fraction = []

    
    def circuit(self, theta): # Creating our circuit
        qc = QuantumCircuit(self.nq)

        for i in range(self.nq):
            qc.h(i)

        for iter in range(self.layers):
            for ia in range(self.na):
                for ir in range(self.na,self.nq):
                    qc.cry(theta[iter * self.na*self.nr + ia*self.nr + (ir-self.na)], ir, ia)

           # if iter < self.layers - 1:
            for n in range(self.na, self.nq, 2):
                if n+1 < self.nq:
                    qc.cx(n, n+1)

            for n in range(self.na+1, self.nq, 2):
                if n+1 < self.nq:
                    qc.cx(n, n+1)
            
            qc.barrier()
        
        qc.measure_all()
        return qc

    def get_solution_distribution(self,solution = 50):
        # Your data here
        optimization_runs = self.optimization_runs
        c_norm = self.cost_fun
        fractions = [element /solution for element in self.fraction]

        # Create a scatter plot
        # plt.figure(figsize=(8, 6))
        scatter = plt.scatter(c_norm, optimization_runs, c=fractions, cmap='viridis', label='compressed RPA')
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

    def sample_solutions(self, params, run, solutions=50):
        temp_cost_fun = []
        temp_fraction = []
        temp_optimization_runs = []
        vqa_circ = self.circuit(params)
        for i in range(solutions):
            job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=100000,  max_parallel_threads=1)
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
            
            self.cost_fun+=temp_cost_fun
            self.fraction+=temp_fraction
            self.optimization_runs+=temp_optimization_runs
    

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
    

    def vqa_cf(self, parameters, number_of_measurements):
        
        
        # Number of qubits = number of variables = length of matrix
        vqa_circuit = self.circuit(parameters)
        #vqa_circuit.draw(output='mpl')
        job = execute(vqa_circuit,Aer.get_backend('qasm_simulator'),shots=number_of_measurements,  max_parallel_threads=1)
        counts = job.result().get_counts(vqa_circuit)
        
        # probabilities = self.calculate_probabilities(counts)
        # obj = self.subset_sum_obj(probabilities)
            
        obj = self.compute_expectation(counts)
        self.temp_cost_evolution.append(obj)
        return obj


    def optimize(self, n_measurements = 100000, number_of_experiments = 10, maxiter = 150):

        opt_vec =[]
        x_vec = []

        sum_time=0
        for k in range(number_of_experiments):
            self.temp_cost_evolution=[]
            initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.na*self.nr)

            print("Begin optimization for hardware efficient CompressedVQA RPA and %.d measurments..." %(n_measurements))
            time1 = time.time()
            opt = minimize(self.vqa_cf,initial_vector, args=(n_measurements), method = 'COBYLA',options={'maxiter': maxiter})
            sum_time += time.time() - time1
            print("Optimization complete. Time taken: %.3fs." %(time.time()-time1))
            print()
            opt_vec.append(opt.fun)
            x_vec.append(opt.x)
            self.sample_solutions(opt.x, k+1)
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
        plt.plot(meancf_mc,linestyle='--',label = 'Compressed VQE RPA mean')

        plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        # plt.show()
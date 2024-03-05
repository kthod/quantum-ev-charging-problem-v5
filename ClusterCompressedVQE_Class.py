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



class split_compressedVQE:

    def __init__(self, qubo_matrix: list, layers: int = 1,group:int =1, na: int = 1) -> None:
        
        self.qubo_matrix = qubo_matrix
       # print(self.qubo_matrix[0])
        #print(self.qubo_matrix[1])
       
        self.layers = layers

        self.group = group

        self.nc = int(len(qubo_matrix)/group)
        #print(self.nc)
        self.nr = int(np.ceil(np.log2(self.nc/na)))
        self.na = na

        self.nq = self.nr + na
        #self.nq = self.nc
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
            ancilla_len = self.na
            probabilities = self.calculate_probabilities(count)
            sol = ""
            for i in range(len(probabilities)):
                ind = np.argmax(probabilities[i,:]) 
                ancilla = f'{ind:0{ancilla_len}b}'
                sol+=ancilla

            sol_list+=sol
            print(f"QUBO SOLUTION IS :{sol}")
        
        return sol_list

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
    
    def compute_expectation(self,group:int,counts: dict) -> float:


            na = self.na
            nc = self.nc

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

            return np.sum(P * self.qubo_matrix[group*self.nc:(group+1)*self.nc,group*self.nc:(group+1)*self.nc])

    def compute_expectation2(self,groupi:int,groupj: int, counts_i: dict, counts_j: dict) -> float:


            na = self.na
            nc = self.nc

            num_registers = int(np.ceil(nc/na))

            register_counts = np.zeros((nc, nc), dtype=int)
            P = np.zeros((nc, nc), dtype=float)
            Pi = np.zeros(nc,dtype = float)
            register_countsi = np.zeros(nc, dtype=int)
            Pj = np.zeros(nc,dtype = float)
            register_countsj = np.zeros(nc, dtype=int)

            for bitstringi, counti in counts_i.items():
            
                auxi = bitstringi[:na]
                regi = int(bitstringi[na:], 2)

                if regi >= num_registers:
                    continue
                for i in range(na):
                   
                    if auxi[i] == '1':
                        Pi[regi * na + i] += counti
                    register_countsi[regi * na + i] += counti

            register_countsi[np.where(register_countsi == 0)] = 1

            Pi /= register_countsi

            for bitstringj, countj in counts_j.items():
            
                auxj = bitstringj[:na]
                regj = int(bitstringj[na:], 2)

                if regj >= num_registers:
                    continue
                for j in range(na):
                   
                    if auxj[j] == '1':
                        Pj[regj * na + j] += countj
                    register_countsj[regj * na + j] += countj

            register_countsj[np.where(register_countsj == 0)] = 1

            Pj /= register_countsj
           
            P = Pi @ Pj.T

            return np.sum(P * self.qubo_matrix[groupi*self.nc:(groupi+1)*self.nc,groupj*self.nc:(groupj+1)*self.nc])
    

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


    def optimize(self, n_measurements = 10000, number_of_experiments = 5, maxiter = 1500):

        opt_vec =[]
        x_vec = []

        sum_time=0
        for k in range(number_of_experiments):
            self.temp_cost_evolution=[]
            initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.nq*self.group)

            print("Begin optimization for hardware efficient VQA and %.d measurments..." %(n_measurements))
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
        plt.plot(meancf_mc,linestyle='--',label = 'Cluster Compressed VQE mean')

        plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        #plt.show()


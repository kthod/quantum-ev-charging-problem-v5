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

class CompressedVQE:

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
        vqa_circ = self.circuit(self.optimal_params)
        job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=10000,  max_parallel_threads=1)
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


    def subset_sum_obj(self,probabilities): # calculates cost function for max cut
        ancilla_len = self.na
        nc = self.nc
        #num_of_registers = int(len(a)/q)
        P =  np.zeros((nc,nc))

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
        # obj =0
        # for i in range(nc):
        #     for j in range(nc):
                
        #         obj+=Q[i][j]*P[i][j]
            
        obj = np.sum(P * Q)   
        
        #print(obj)      
        return obj

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


    def optimize(self, n_measurements = 10000, number_of_experiments = 10, maxiter = 150):

        opt_vec =[]
        x_vec = []

        sum_time=0
        for k in range(number_of_experiments):
            self.temp_cost_evolution=[]
            initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.nq)

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
        plt.plot(meancf_mc,linestyle='--',label = 'Compressed VQE mean')

        plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        # plt.show()








class VQE:

    def __init__(self, qubo_matrix: np.array, layers: int = 1) -> None:
        
        self.qubo_matrix = qubo_matrix
        self.layers = layers

        

        self.nc = len(qubo_matrix)
   

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
        vqa_circ = self.circuit(self.optimal_params)
        job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=10000,  max_parallel_threads=1)
        count = job.result().get_counts(vqa_circ)
        sol = max(count, key=count.get)

        print(f"QUBO SOLUTION IS :{sol}")
        return sol
    # Additional functions to calculate the cost function value so we can pass it
    # to our optimizer (COBYLA).


    def subset_sum_obj(self,probabilities): # calculates cost function for max cut
        ancilla_len = self.na
        nc = self.nc
        #num_of_registers = int(len(a)/q)
        P =  np.zeros((nc,nc))

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
        # obj =0
        # for i in range(nc):
        #     for j in range(nc):
                
        #         obj+=Q[i][j]*P[i][j]
            
        obj = np.sum(P * Q)   
        
        #print(obj)      
        return obj

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


            sum_en = 0
            total_counts = 0
            for bitstr, count in counts.items():

                bitstring = np.array([int(x) for x in bitstr])

                sum_en += count*(bitstring.T @ self.qubo_matrix @ bitstring)
                total_counts += count


            return sum_en/total_counts
    

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


    def optimize(self, n_measurements = 10000, number_of_experiments = 5, maxiter = 400):

        opt_vec =[]
        x_vec = []

        sum_time=0
        for k in range(number_of_experiments):
            self.temp_cost_evolution=[]
            initial_vector = np.random.uniform(0,2*np.pi, self.layers*self.nq)

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
        plt.plot(meancf_mc,linestyle='--',label = 'VQE mean')

        plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        # plt.show()

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
        sol_list = []
        for i in range(self.group):
            vqa_circ = self.circuit(self.optimal_params[i*self.nq*self.layers:(i+1)*self.nq*self.layers])
            job = execute(vqa_circ,Aer.get_backend('qasm_simulator'),shots=10000,  max_parallel_threads=1)
            count = job.result().get_counts(vqa_circ)
            sol = max(count, key=count.get)
            sol_list.append(sol)
            print(f"QUBO SOLUTION IS :{sol}")
        
        return sol_list

    # Additional functions to calculate the cost function value so we can pass it
    # to our optimizer (COBYLA).


    def subset_sum_obj(self,probabilities): # calculates cost function for max cut
        ancilla_len = self.na
        nc = self.nc
        #num_of_registers = int(len(a)/q)
        P =  np.zeros((nc,nc))

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
        # obj =0
        # for i in range(nc):
        #     for j in range(nc):
                
        #         obj+=Q[i][j]*P[i][j]
            
        obj = np.sum(P * Q)   
        
        #print(obj)      
        return obj

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
        plt.plot(meancf_mc,linestyle='--',label = 'ClusterVQE mean')

        plt.fill_between(range(maxiter), mincf_mc, maxcf_mc, alpha=0.5)
        #plt.plot(cflistVQA)
        #plt.ylim(-4,0)
        # plt.title("Cost function with iterations, hardware efficient VQA\n minimal encoding")
        # plt.ylabel("Cost function value")
        # plt.xlabel("Number of COBYLA iterations")

        # plt.show()



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
        sol_list = []
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

            sol_list.append(sol)
            print(f"QUBO SOLUTION IS :{sol}")
        
        return sol_list

    # Additional functions to calculate the cost function value so we can pass it
    # to our optimizer (COBYLA).


    def subset_sum_obj(self,probabilities): # calculates cost function for max cut
        ancilla_len = self.na
        nc = self.nc
        #num_of_registers = int(len(a)/q)
        P =  np.zeros((nc,nc))

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
        # obj =0
        # for i in range(nc):
        #     for j in range(nc):
                
        #         obj+=Q[i][j]*P[i][j]
            
        obj = np.sum(P * Q)   
        
        #print(obj)      
        return obj

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



def get_qubomat(a,S):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    matrix = np.zeros((len(a),len(a)))
    nvar = len(matrix)
    
    for i in range(len(a)):
        for j in range(i,len(a)):
            if i == j:
                matrix[i][i] = (a[i]**2) - 2*S*a[i]
            else:
                matrix[i][j] = 2*a[i]*a[j]
    return matrix


a=[2,2,2,2,2,2,2,2,2,2]
S=10
epsilon = np.array([26880,26880,26880,7680,23040,23040,7680,7680])
de = np.array([3,3,3,1,3,3,1,1])


Q = []

def get_qubo2(eps,delta,Horizon,V,DT):
    
    d = np.ones(2*Horizon)

# Then, change the elements after the first k elements to 1000
    d[2*delta:] = 1000
    print(d)
    p = np.array([2 ** (i % 2) for i in range(2*Horizon)])
    #print(p)
    Q = 256*V**2*DT*(d.T*p.T) @ (d*p) - 16*2*V*DT*eps*np.diag(d*p)
#64*V**2*DT*np.diag(d**2*p**2)
    return Q

def get_qubomat3(eps,delta,Horizon,V,DT):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    d = np.ones(2*Horizon)
    #print(eps)
# Then, change the elements after the first k elements to 1000
    d[2*delta:] = 0
    #print(d)
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    p = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(d),len(d)))
    nvar = len(matrix)
    
    for i in range(len(d)):
        for j in range(i,len(d)):
            if i == j:
                matrix[i][i] = 256*(V**2)*(DT**2) * ((p[i]**2)*(d[i]**2)) - 16*2*eps*V*DT*d[i]*p[i]# + 256*p[i]**2
            else:
                matrix[i][j] = 256*2*(V**2)*(DT**2) *p[i]*p[j]*d[i]*d[j]
    return matrix


def get_qubomat4(evi,evj,deltai,deltaj,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    di = np.ones(2*Horizon)
    dj = np.ones(2*Horizon)
# Then, change the elements after the first k elements to 1000
    di[2*deltai:] = 0
    dj[2*deltaj:] = 0
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(di),len(di)))
    nvar = len(matrix)
    if evi==evj:
        for i in range(0,len(di),2):
 
            matrix[i:(i+2),i:(i+2)] =  256*np.array([[(pi[i]**2)*di[i]**2,2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],[0,(pi[i+1]**2)*di[i+1]**2]]) 
    else:
        for i in range(0,len(di),2):
            matrix[i:(i+2),i:(i+2)] =2*256*np.array([[(pi[i]**2)*di[i]*dj[i],(pi[i]*pi[i+1])*(di[i]*dj[i+1])],[(pi[i]*pi[i+1])*(dj[i]*di[i+1]),(pi[i+1]**2)*di[i+1]*dj[i+1]]])           
    return matrix
#Q = get_qubomat(a,S)

# Q = np.array([[1,1,0,0],
#      [0,1,0,0],
#      [0,0,1,1],
#      [0,0,0,1]]
# )
# Q1 = Q[0:5,0:5]
# Q2 = Q[5:10,5:10]
# Q3 = Q[0:5,5:10]
# print(Q)


evs = 4
V=240
DT = 1
Horizon = 4
#def get_qubomat4():
Q  = np.zeros((evs*Horizon*2,evs*Horizon*2))
Q1  = np.zeros((evs*Horizon*2,evs*Horizon*2))
#Q = get_qubomat3(11520,3,Horizon,V,DT)
for ev in range(evs):
    Q[ev*Horizon*2:(ev+1)*Horizon*2,ev*Horizon*2:(ev+1)*Horizon*2] = get_qubomat3(epsilon[ev],de[ev],Horizon,V,DT)

for evi in range(evs):
    for evj in range(evi,evs):
        Q1[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = get_qubomat4(evi,evj,de[evi],de[evj],Horizon)
        #print(Q1[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2])


#print(Q1)

Q = Q + Q1
# inst = VQE(Q,2)
# inst.optimize()
# inst.show_solution()
# for i in range(evs*(evs-1)/2):
#     Q.append() 

def ret_schedule(solution):
    sched = np.zeros((evs,Horizon))
    for i in range(evs):
        sol = solution[i]
        #bitstring = np.array([int(x) for x in sol])
        for j in range(Horizon):
            if j<de[i]:
                sched[i,j] = 16*int(sol[2*j:2*j+2],2)
            else:
                sched[i,j] = 0

    
    print(sched)

def reshape_solution(sol):
    solution = []
    for i in range(evs):
        solution.append(sol[i*2*Horizon:(i+1)*2*Horizon])
        
    return solution
  
#print(Q)
inst = CompressedVQE(Q,2)
inst.optimize(number_of_experiments = 10,maxiter=300)
solution = inst.show_solution()

solution = reshape_solution(solution)
ret_schedule(solution)

inst = split_compressedVQE(Q,2,group=evs)
inst.optimize(number_of_experiments = 10,maxiter=300)
solution = inst.show_solution()

ret_schedule(solution)

inst = splitVQE(Q,2,evs)
inst.optimize(number_of_experiments = 5,maxiter=300)
solution = inst.show_solution()

ret_schedule(solution)

inst = VQE(Q,2)
inst.optimize(number_of_experiments = 3,maxiter=300)
solution = inst.show_solution()

solution = reshape_solution(solution)
ret_schedule(solution)

plt.legend()
plt.show()


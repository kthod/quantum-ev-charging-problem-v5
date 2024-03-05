from qiskit import *
from qiskit.circuit import Parameter
import cvxpy as cp
from scipy.optimize import minimize
from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#import dimod

from abc import ABC, abstractclassmethod

class OptimizationAlgorithm(ABC):

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1, evs: int = 0, horizon : int = 1) -> None:
        self.qubo_matrix = qubo_matrix
        self.layers = layers
        self.nq = 0
        self.num_of_EV = evs
        self.horizon = horizon
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

class CompressedVQE(OptimizationAlgorithm):

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1, na: int = 1,evs: int = 0, horizon : int = 1,
                  timestep: float = 1,epsilon : np.array =[], d :np.array = [], total_rate_limit :float = 300.0) -> None:
        
        if na == len(qubo_matrix):
            raise ValueError("For full encoding use VQE instead.")
        
        super().__init__(qubo_matrix, layers,evs,horizon)

        nc = len(qubo_matrix)
        nr = int(np.ceil(np.log2(nc/na)))
        self.na = na
        self.timestep = timestep
        self.nq = nr + na
        self.nq = int(np.ceil(np.log2(evs)))
        self.algorithm = "Compressed VQE"
        self.epsilon = epsilon
        self.d = d
        self.total_rate_limit = total_rate_limit
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
        
        #qc.measure_all()
        return qc
        
    def optimize(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
        voltage =240
        def cost_function(statevector):
            
            rates = self.total_rate_limit*statevector
            def quick_charge(rates):
                optimization_horizon = self.horizon

                c = np.array(
                    [
                        (optimization_horizon - t) / optimization_horizon
                        for t in range(optimization_horizon)
                    ]
                )
                return c @ np.sum(rates, axis=0)

            def equal_share(rates):
                

                return -np.sum(np.sum(rates, axis = 0)**2)
            
            def non_completion(rates):
                energy = self.timestep*voltage*rates
                s = 0
                for i in range(self.num_of_EV):

                    s -= np.abs(sum(energy[i,:self.d[i]]) - self.epsilon[i])
                return s
            #print(quick_charge(rates))
            return 10*quick_charge(rates) +100*non_completion(rates)+ 1*equal_share(rates)

        def objective():
            backend = Aer.get_backend('aer_simulator')

            def execute_circ(theta):
                squared_magnitudes_list = []
                for i in range(self.horizon):
                    qc = self.build_circuit(theta[i*self.nq*self.layers: (i+1)*self.nq*self.layers])
                    qc.save_statevector()
                    backend.set_options(method='statevector')
                    statevector_job = execute(qc, backend)
                    statevector_result = statevector_job.result()
                    statevector = statevector_result.get_statevector().data
                    statevector = statevector[:self.num_of_EV]
                    squared_magnitudes_list.append(np.real(statevector)**2 + np.imag(statevector)**2)
                
                squared_magnitudes = np.column_stack(squared_magnitudes_list)
                #counts = execute(qc, backend, shots=shots).result().get_counts()
                #print(self.num_of_EV*self.horizon)
                exp = cost_function(squared_magnitudes)
                self.temp_cost_evolution.append(-exp)
                return -exp 
            
            return execute_circ
        opt_vec =[]
        x_vec = []
        for _ in range(experiments):

            self.temp_cost_evolution = list()

            obj = objective()
            theta = np.random.uniform(low=0, high=np.pi, size=self.layers*self.nq*self.horizon)
            result = minimize(obj, theta, method="COBYLA", options={"maxiter": maxiter})

            self.cost_evolution.append(np.asarray((self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution)))))
            opt_vec.append(result.fun)
            x_vec.append(result.x)
            print('Experiment:', (_+1), "min:", result.fun)
        self.optimal_cost = np.min(opt_vec)
        ind = np.argmin(opt_vec)
        self.optimal_answer = x_vec[ind]
            # Execute the final quantum circuit one more time using the optimized parameters.
        backend = Aer.get_backend('aer_simulator')
        squared_magnitudes_list = []
        for i in range(self.horizon):
            qc = self.build_circuit(self.optimal_answer[i*self.nq*self.layers: (i+1)*self.nq*self.layers])
            qc.save_statevector()
            backend.set_options(method='statevector')
            statevector_job = execute(qc, backend)
            statevector_result = statevector_job.result()
            statevector = statevector_result.get_statevector().data
            
            statevector = statevector[:self.num_of_EV]
            #print(np.real(statevector)**2 + np.imag(statevector)**2)
            squared_magnitudes_list.append(np.real(statevector)**2 + np.imag(statevector)**2)
            
            squared_magnitudes = np.column_stack(squared_magnitudes_list)
        # print(np.sum(squared_magnitudes))
        
        # print(squared_magnitudes)
        # print(np.sum(self.total_rate_limit*voltage*squared_magnitudes,axis = 1))
        # print(np.sum(squared_magnitudes,axis = 0))
        return self.total_rate_limit*squared_magnitudes


class CompressedVQE2(OptimizationAlgorithm):

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1, na: int = 1,evs: int = 0, horizon : int = 1,
                  timestep: float = 1,epsilon : np.array =[], d :np.array = [], total_rate_limit :float = 300.0) -> None:
        
        if na == len(qubo_matrix):
            raise ValueError("For full encoding use VQE instead.")
        
        super().__init__(qubo_matrix, layers,evs,horizon)

        nc = len(qubo_matrix)
        nr = int(np.ceil(np.log2(nc/na)))
        self.na = na
        self.timestep = timestep
        self.nq = nr + na
        self.nq = int(np.ceil(np.log2(evs*horizon)))
        print(self.nq)
        self.algorithm = "Compressed VQE"
        self.epsilon = epsilon
        self.d = d
        self.total_horizon_rate_limit = horizon*total_rate_limit
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
        
        #qc.measure_all()
        return qc
        
    def optimize(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
        voltage =240
        def cost_function(statevector):
            
            rates = self.total_horizon_rate_limit*statevector
            def quick_charge(rates):
                optimization_horizon = self.horizon

                c = np.array(
                    [
                        (optimization_horizon - t) / optimization_horizon
                        for t in range(optimization_horizon)
                    ]
                )
                return c @ np.sum(rates, axis=0)

            def equal_share(rates):
                

                return -np.sum(np.sum(rates, axis = 0)**2)
            
            def non_completion(rates):
                energy = self.timestep*voltage*rates
                s = 0
                for i in range(self.num_of_EV):

                    s -= np.abs(sum(energy[i,:self.d[i]]) - self.epsilon[i])
                return s
            #print(quick_charge(rates))
            return 10*quick_charge(rates) +100*non_completion(rates)+ 1*equal_share(rates)

        def objective():
            backend = Aer.get_backend('aer_simulator')

            def execute_circ(theta):
                
                qc = self.build_circuit(theta)
                qc.save_statevector()
                backend.set_options(method='statevector')
                statevector_job = execute(qc, backend)
                statevector_result = statevector_job.result()
                statevector = statevector_result.get_statevector().data
                statevector = statevector[:self.num_of_EV*self.horizon]
                squared_magnitudes = np.real(statevector)**2 + np.imag(statevector)**2
                
                squared_magnitudes = np.reshape(squared_magnitudes,(self.num_of_EV,self.horizon))
                #counts = execute(qc, backend, shots=shots).result().get_counts()
                #print(self.num_of_EV*self.horizon)
                exp = cost_function(squared_magnitudes)
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
            opt_vec.append(result.fun)
            x_vec.append(result.x)
            print('Experiment:', (_+1), "min:", result.fun)
        self.optimal_cost = np.min(opt_vec)
        ind = np.argmin(opt_vec)
        self.optimal_answer = x_vec[ind]
            # Execute the final quantum circuit one more time using the optimized parameters.
        backend = Aer.get_backend('aer_simulator')
        squared_magnitudes_list = []
        
        qc = self.build_circuit(self.optimal_answer)
        qc.save_statevector()
        backend.set_options(method='statevector')
        statevector_job = execute(qc, backend)
        statevector_result = statevector_job.result()
        statevector = statevector_result.get_statevector().data
        
        statevector = statevector[:self.num_of_EV*self.horizon]
        #print(np.real(statevector)**2 + np.imag(statevector)**2)
        squared_magnitudes = np.real(statevector)**2 + np.imag(statevector)**2
        
        squared_magnitudes = np.reshape(squared_magnitudes,(self.num_of_EV,self.horizon))
        # print(np.sum(squared_magnitudes))
        
        # print(squared_magnitudes)
        # print(np.sum(self.total_rate_limit*voltage*squared_magnitudes,axis = 1))
        # print(np.sum(squared_magnitudes,axis = 0))
        return self.total_horizon_rate_limit*squared_magnitudes

class CompressedVQE3(OptimizationAlgorithm):

    def __init__(self, qubo_matrix: np.ndarray, layers: int = 1, na: int = 1,evs: int = 0, horizon : int = 1,
                  timestep: float = 1,epsilon : np.array =[], d :np.array = [], total_rate_limit :float = 300.0, voltage = 240) -> None:
        
        if na == len(qubo_matrix):
            raise ValueError("For full encoding use VQE instead.")
        
        super().__init__(qubo_matrix, layers,evs,horizon)

        nc = len(qubo_matrix)
        nr = int(np.ceil(np.log2(nc/na)))
        self.na = na
        self.timestep = timestep
        self.nq = nr + na
        self.nq = int(np.ceil(np.log2(horizon)))
        self.algorithm = "Compressed VQE3"
        self.epsilon = epsilon
        self.d = d
        self.total_rate_limit = total_rate_limit
        self.voltage = voltage
        demand = sum(epsilon)/voltage
        self.rate_scaling =total_rate_limit*horizon/demand* epsilon/voltage
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
        
        #qc.measure_all()
        return qc
        
    def optimize(self, maxiter: int = 1000, shots: int = 10000, experiments: int = 1) -> str:
        voltage =240
        def cost_function(statevector):
            
            rates = statevector
            def quick_charge(rates):
                optimization_horizon = self.horizon

                c = np.array(
                    [
                        (optimization_horizon - t) / optimization_horizon
                        for t in range(optimization_horizon)
                    ]
                )
                return c @ np.sum(rates, axis=0)

            def equal_share(rates):
                

                return -np.sum(np.sum(rates, axis = 0)**2)
            
            def non_completion(rates):
                energy = self.timestep*voltage*rates
                s = 0
                for i in range(self.num_of_EV):

                    s -= np.abs(sum(energy[i,:self.d[i]]) - self.epsilon[i])
                return s
            #print(quick_charge(rates))
            return 10*quick_charge(rates) +100*non_completion(rates)+ 1*equal_share(rates)

        def objective():
            backend = Aer.get_backend('aer_simulator')

            def execute_circ(theta):
                squared_magnitudes_list = []
                for i in range(self.num_of_EV):
                    qc = self.build_circuit(theta[i*self.nq*self.layers: (i+1)*self.nq*self.layers])
                    qc.save_statevector()
                    backend.set_options(method='statevector')
                    statevector_job = execute(qc, backend)
                    statevector_result = statevector_job.result()
                    statevector = statevector_result.get_statevector().data
                    statevector = statevector[:self.horizon]
                    squared_magnitudes_list.append(np.real(statevector)**2 + np.imag(statevector)**2)
                
                squared_magnitudes = np.row_stack(squared_magnitudes_list)
                #counts = execute(qc, backend, shots=shots).result().get_counts()
                #print(self.num_of_EV*self.horizon)
                exp = cost_function(self.rate_scaling.reshape(-1,1)*squared_magnitudes)
                self.temp_cost_evolution.append(-exp)
                return -exp 
            
            return execute_circ
        opt_vec =[]
        x_vec = []
        for _ in range(experiments):

            self.temp_cost_evolution = list()

            obj = objective()
            theta = np.random.uniform(low=0, high=np.pi, size=self.layers*self.nq*self.num_of_EV)
            result = minimize(obj, theta, method="COBYLA", options={"maxiter": maxiter})

            self.cost_evolution.append(np.asarray((self.temp_cost_evolution + [self.temp_cost_evolution[-1]] * (maxiter - len(self.temp_cost_evolution)))))
            opt_vec.append(result.fun)
            x_vec.append(result.x)
            print('Experiment:', (_+1), "min:", result.fun)
        self.optimal_cost = np.min(opt_vec)
        ind = np.argmin(opt_vec)
        self.optimal_answer = x_vec[ind]
            # Execute the final quantum circuit one more time using the optimized parameters.
        backend = Aer.get_backend('aer_simulator')
        squared_magnitudes_list = []
        for i in range(self.num_of_EV):
            qc = self.build_circuit(theta[i*self.nq*self.layers: (i+1)*self.nq*self.layers])
            qc.save_statevector()
            backend.set_options(method='statevector')
            statevector_job = execute(qc, backend)
            statevector_result = statevector_job.result()
            statevector = statevector_result.get_statevector().data
            statevector = statevector[:self.horizon]
            squared_magnitudes_list.append(np.real(statevector)**2 + np.imag(statevector)**2)
            print(sum((np.real(statevector)**2 + np.imag(statevector)**2)))
        squared_magnitudes = np.row_stack(squared_magnitudes_list)
        # print(np.sum(squared_magnitudes))
        
        # print(squared_magnitudes)
        # print(np.sum(self.total_rate_limit*voltage*squared_magnitudes,axis = 1))
        # print(np.sum(squared_magnitudes,axis = 0))
        return self.rate_scaling.reshape(-1,1)*squared_magnitudes






class MonitoredSimulatedAnnealing:

    def __init__(self) -> None:
        pass

    def anneal(self, runtime: int) -> str:
        pass






    # def calculate_probabilities(counts,nc,na):

    #         probabilities = np.zeros((int(nc/na), 2**na))
    #         # counter_01s = np.zeros(int(nc/2))
    #         # counter_10s = np.zeros(int(nc/2))
    #         # counter_11s = np.zeros(int(nc/2))
    #         num_registers = int(np.ceil(nc/na))
    #         counter = np.zeros(int(nc/na))
    #         for bitstr, count in counts.items():
                
    #             ancilla = int(bitstr[0:na],2)
                
    #             register = int(bitstr[na:],2)
                
    #             if register >= num_registers:
    #                 continue
    #             counter[register]+=count
    #             probabilities[register][ancilla]+=count
                
    #         if np.any(counter==0):
    #             ind = np.where(counter == 0)[0]

    #             counter[ind] = 2
    #             probabilities[ind,:] = 1

    #         for i in range(len(counter)):
    #             probabilities[i, :] /= counter[i]
    #             # print(probabilities)
    #             # ow_sums = np.sum(probabilities, axis=1)
    #             # print(ow_sums)
    #         return probabilities
        
    #     def compute_expectation(counts: dict) -> float:
    #         # na = self.na
    #         # nc = len(self.qubo_matrix)
    #         # P =  np.zeros((nc, nc)) 

    #         # probabilities = calculate_probabilities(counts,nc,na)           
    #         # for i in range(nc):
    #         #     anc_index_i = i%na
    #         #     reg_index_i = i//na                #print("start")
    #         #     for ancilla_int in range(2**na):
    #         #         ancilla = f'{ancilla_int:0{na}b}'
    #         #         #print(ancilla)
    #         #         if ancilla[anc_index_i]=='1':
    #         #             #print(f"i {i} j {j}")
    #         #             P[i][i] += probabilities[reg_index_i][ancilla_int]            
    #         # for i in range(nc-1):
    #         #     for j in range(i+1,nc):                    
    #         #         anc_index_i = i % na
    #         #         reg_index_i = i // na
    #         #         anc_index_j = j % na
    #         #         reg_index_j = j // na                    
    #         #         if reg_index_i == reg_index_j:
    #         #             #print("start")
    #         #             for ancilla_int in range(2**na):
    #         #                 ancilla = f'{ancilla_int:0{na}b}'
    #         #                 #print(ancilla)
    #         #                 if ancilla[anc_index_i]=='1' and ancilla[anc_index_j]=='1':
    #         #                     #print(f"i {i} j {j}")
    #         #                     P[i][j] += probabilities[reg_index_i][ancilla_int]                    
    #         #         else:
                      
    #         #             P[i][j] = P[i][i] * P[j][j]            # x = np.array(x)
    #         # # a= np.array(a)
    #         # # print(a)
    #         # # print(x)
    #         # obj = np.sum(self.qubo_matrix * P)
    #         # return obj


    #         ################################################
    #         na = self.na
    #         nc = len(self.qubo_matrix)

    #         num_registers = int(np.ceil(nc/na))

    #         register_counts = np.zeros((nc, nc), dtype=int)
    #         P = np.zeros((nc, nc), dtype=float)

    #         for bitstring, count in counts.items():

    #             aux = bitstring[:na]
    #             reg = int(bitstring[na:], 2)

    #             if reg >= num_registers:
    #                 continue

    #             for i in range(na):
    #                 for j in range(i, na):
    #                     if aux[i] == '1' and aux[j] == '1':
    #                         P[reg * na + i][reg * na + j] += count
    #                     register_counts[reg * na + i][reg * na + j] += count

    #         register_counts[np.where(register_counts == 0)] = 1

    #         P /= register_counts

    #         for i in range(nc-1):
    #             for j in range(i+1, nc):
    #                 reg_index_i = i // na
    #                 reg_index_j = j // na

    #                 if reg_index_i != reg_index_j:
    #                     P[i][j] = P[i][i] * P[j][j]

    #         return np.sum(P * self.qubo_matrix)
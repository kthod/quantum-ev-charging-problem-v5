import sys
import os
file_directory = os.path.dirname(os.path.abspath(__file__))
pthh = file_directory+r"\dist"
print(pthh)
sys.path.append(file_directory+r"\dist")
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
from VQE_Class import*
from CompressedVQE_Class_obfus import*
from CompressedVQE_RPA_Class import*
from ClusterVQE_Class import*
from ClusterCompressedVQE_Class import*
import matplotlib.colors as mcolors


class Quantum_MPC():
    """
    A class for Quantum Model Predictive Control (MPC) using variational quantum algorithms.
    
    Attributes:
        epsilon (array): Array of epsilon values representing energy demands of each EV.
        de (array): Array of integers indicating the time duration demand of each EV.
        C (float): Network's Power Capacity.
        Horizon (int): The horizon over which the optimization is performed.
        DT (float): The time step size or duration.
        inst: Instance of the optimization algorithm used.
    """
    def __init__(self,epsilon , de, C, Horizon, DT, algorithm = "CompressedVQE", layers = 2) -> None:
        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C
        self.Horizon = Horizon
        self.DT = DT
        self.inst = None
        Q = self.get_qubomat(self.epsilon , self.de, self.C, self.Horizon, self.DT)

        if algorithm == "CompressedVQE":
            inst = CompressedVQE(Q,layers,na=2)
            #inst.optimize(n_measurements = n_measurements,number_of_experiments = number_of_experiments,maxiter=maxiter)
        else:
            inst = VQE(Q,layers)
            #inst.optimize(n_measurements = n_measurements,number_of_experiments = number_of_experiments,maxiter=maxiter)

        self.inst = inst

    def get_qubomat_NC(self, eps, delta, Horizon, V, DT):
        """
        Generates a block 2*Horizon x 2*Horizon of the QUBO matrix  relates to one EV for Non-Completed demands terms in the optimization problem.

        Parameters:
            eps (float): Energy demand or cost for a specific EV.
            delta (int): Time Duration Demand for a specific EV.
            Horizon (int): Optimization horizon.
            V (float): Voltage.
            DT (float): Time step size .

        Returns:
            numpy.ndarray: A QUBO matrix for Non-Completion terms.
        """
        # Initialize a ones array with length 2*Horizon to represent binary variables for each time step.
        d = np.ones(2*Horizon)
        # Set elements beyond the delta index to 0 to represent demand constraints.
        d[2*delta:] = 0
        
        p = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
        matrix = np.zeros((len(d), len(d)))
        
        
        for i in range(len(d)):
            for j in range(i, len(d)):
                if i == j:
                    matrix[i][i] = 256*(V**2)*(DT**2) * ((p[i]**2)*(d[i]**2)) - 16*2*eps*V*DT*d[i]*p[i]
                else:
                    matrix[i][j] = 256*2*(V**2)*(DT**2) * p[i]*p[j]*d[i]*d[j]
        return matrix



    def get_qubomat_LV(self, evi, evj, deltai, deltaj, Horizon):
        """
        Generates a block 2*Horizon x 2*Horizon of the QUBO matrix which captures the relationship of two EVs i and j
        regarding their Load variationn demands terms in the optimization problem.

        Parameters:
            evi (int): EV i.
            evi (int): EV i.
            deltai (int): Time Duration Demand for EV i.
            deltaj (int): Time Duration Demand for EV j.
            Horizon (int): Optimization horizon.
            DT (float): Time step size .

        Returns:
            numpy.ndarray: A QUBO matrix for non-coupled terms.
        """


        di = np.ones(2*Horizon)
        dj = np.ones(2*Horizon)
        
  
        di[2*deltai:] = 0
        dj[2*deltaj:] = 0

        
        pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])

        
        # Initialize the QUBO matrix with zeros. It will be populated based on the interaction between the  evs.
        matrix = np.zeros((len(di), len(di)))

        # If we are considering the same evs (self-interaction),
        # populate the diagonal blocks of the matrix accordingly.
        if evi == evj:
            for i in range(0, len(di), 2):
                matrix[i:(i+2), i:(i+2)] = 256*np.array([
                    [(pi[i]**2)*di[i]**2, 2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],
                    [0, (pi[i+1]**2)*di[i+1]**2]
                ])
        else:
            # For interactions between different evs,
            # populate the matrix with cross-terms that represent these interactions.
            for i in range(0, len(di), 2):
                matrix[i:(i+2), i:(i+2)] = 2*256*np.array([
                    [(pi[i]**2)*di[i]*dj[i], (pi[i]*pi[i+1])*(di[i]*dj[i+1])],
                    [(pi[i]*pi[i+1])*(dj[i]*di[i+1]), (pi[i+1]**2)*di[i+1]*dj[i+1]]
                ])
        
        # Return the constructed QUBO matrix.
        return matrix


    def get_qubomat_P1(self, evi, evj, deltai, deltaj, Horizon, V, C):
        """
        Generates the first part of the QUBO matrix related to inequality constraint as square penalties. This part has to do with the interaction
        between the original problems optimization variables and the slack variable
        Calculates a block 2*Horizon x 2*Horizon of the QUBO matrix which captures the relationship of two EVs i and j
       

        Parameters:
            evi (int): EV i.
            evi (int): EV i.
            deltai (int): Time Duration Demand for EV i.
            deltaj (int): Time Duration Demand for EV j.
            Horizon (int): Optimization horizon.
            DT (float): Time step size .

            V (float): Voltage
            C (float): Network's Power Capacity

        Returns:
            numpy.ndarray: A QUBO matrix 
        """

        # Initialize arrays for binary variables representing whether energy is consumed at each time step.
        di = np.ones(2*Horizon)
        dj = np.ones(2*Horizon)
        
        # Set binary variables to 0 beyond the demand limits to represent constraints.
        di[2*deltai:] = 0
        dj[2*deltaj:] = 0

        pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
        

        # Initialize the QUBO matrix for penalty terms.
        matrix = np.zeros((len(di), len(di)))

        # If evaluating penalty terms for the same ev:
        if evi == evj:
            for i in range(0, len(di), 2):
    
                matrix[i:(i+2), i:(i+2)] = np.array([
                    [256*V**2*(pi[i]**2)*di[i]**2 - 2*C*V*16*pi[i]*di[i], 256*V**2*2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],
                    [0, 256*V**2*(pi[i+1]**2)*di[i+1]**2 - 2*C*V*16*pi[i+1]*di[i+1]]
                ])
        else:
            # For different evs, calculate cross-terms representing interactions.
            for i in range(0, len(di), 2):
                matrix[i:(i+2), i:(i+2)] = 2*256*V**2*np.array([
                    [(pi[i]**2)*di[i]*dj[i], (pi[i]*pi[i+1])*(di[i]*dj[i+1])],
                    [(pi[i]*pi[i+1])*(dj[i]*di[i+1]), (pi[i+1]**2)*di[i+1]*dj[i+1]]
                ])
        
        # Return the constructed matrix.
        return matrix


    def get_qubomat_P2(self, V, C, Horizon):
        """
        Generates the second part of the QUBO matrix related to inequality constraint as square penalties. This part has to do with the interaction
        between the slack variables themselves
        Calculates a block 4*Horizon x 4*Horizon of the QUBO matrix which captures the reationship between the slack variables
       

        Parameters:
            Horizon (int): Optimization horizon.
            V (float): Voltage
            C (float): Network's Power Capacity

        Returns:
            numpy.ndarray: A QUBO matrix 
        """


        # pi  are coefficient array influencing how each variable contributes to the penalty.
        # pi's values are determined with a modulus that creates a repeating pattern of 1, 2, 4.
        pi = np.array([2 ** ((i+3) % 3) for i in range(4*Horizon)])

        matrix = np.zeros((4*Horizon, 4*Horizon))

        # Loop through the matrix in blocks of 4 to assign penalty values based on the pi array.
        for i in range(0, 4*Horizon, 4):
            # Fill in a 4x4 block within the matrix for each iteration, creating a pattern that reflects
            # the penalty structure being modeled. This includes both diagonal terms representing individual
            # penalty contributions and off-diagonal terms representing interactions between different variables.
            matrix[i:(i+4), i:(i+4)] = np.array([
                [256*V**2*(pi[i]**2) - 2*C*V*16*pi[i], 256*V**2*2*(pi[i]*pi[i+1]), 256*V**2*2*(pi[i]*pi[i+2]), 256*V**2*2*(pi[i]*pi[i+3])],
                [0, 256*V**2*(pi[i+1]**2) - 2*C*V*16*pi[i+1], 256*V**2*2*(pi[i+1]*pi[i+2]), 256*V**2*2*(pi[i+1]*pi[i+3])],
                [0, 0, 256*V**2*(pi[i+2]**2) - 2*C*V*16*pi[i+2], 256*V**2*2*(pi[i+2]*pi[i+3])],
                [0, 0, 0, 256*V**2*(pi[i+3]**2) - 2*C*V*16*pi[i+3]]
            ])

        return matrix


    def get_qubomat_P3(self,delta, V,Horizon):
        """
        Generates the third part of the QUBO matrix related to inequality constraint as square penalties. This part has to do with the interaction
        between the slack variables and the problem's variable
        Calculates a block 2*Horizon x 4*Horizon of the QUBO matrix which captures the reationship between the slack variables and 1 EV
       

        Parameters:
            delta (int): Time duration demand of the EV
            Horizon (int): Optimization horizon.
            V (float): Voltage
 

        Returns:
            numpy.ndarray: A QUBO matrix 
        """

        d = np.ones(2*Horizon)
        

        d[2*delta:] = 0
        
        pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
        pj = np.array([2 ** ((i+3) % 4) for i in range(4*Horizon)])
        matrix = np.zeros((len(d),4*Horizon))
    

    
        for i in range(0, Horizon):
            matrix[i*2:(i+1)*2,i*4:(i+1)*4] = 2*256*V**2*np.array([[pi[i]*d[i]*pj[i], pi[i]*d[i]*pj[i+1], pi[i]*d[i]*pj[i+2], pi[i]*d[i]*pj[i+3]],
                                                                    [pi[i+1]*d[i+1]*pj[i], pi[i+1]*d[i+1]*pj[i+1], pi[i+1]*d[i+1]*pj[i+2], pi[i+1]*d[i+1]*pj[i+3]]])            
    
        return matrix

    def get_qubomat_QC(self,T,evs):
        """
        Generates the QUBO matrix for Quick-Charging cost function.

        Parameters:
            evs (int): Number of active EVs
            T (int): Horizon

        Returns:
            numpy.ndarray: A QUBO matrix for Non-Completion terms.
        """
        coeff = evs* [(T-i//2)/T for i in range(2*T)]
        
        return -np.diag(coeff)

    def ret_schedule(self,solution,evs, Horizon,de):
        """
        Converts a binary solution from the optimization algorithm into a schedule format.
        This method maps the binary representation of the solution into a more readable and
        interpretable schedule of values for each entity over the optimization horizon.

        Parameters:
            solution (list of str): The solution from the optimization algorithm, where each string
                                    represents the binary encoding of the solution for each entity.
            evs (int): The number of evs
            Horizon (int): The optimization horizon
            de (list of int): A list containing the time duration demand for each ev.

        Returns:
            numpy.ndarray: A schedule matrix where each row represents an entity and each column represents
                        a time step within the horizon. The values in the matrix indicate the level of
                        activity or allocation for each entity at each time step.
        """
        sched = np.zeros((evs,Horizon))
        for i in range(evs):
            sol = solution[i]
            #bitstring = np.array([int(x) for x in sol])
            for j in range(Horizon):
                if j<de[i]:
                    sched[i,j] = 16*int(sol[2*j:2*j+2],2)
                else:
                    sched[i,j] = 0

        
        return sched

    def reshape_solution(self,sol,evs, Horizon):
        """
        Reshapes the flat solution string into a list of strings, where each string represents the solution 
        for an individual entity over the entire optimization horizon. This method is particularly useful 
        when the solution from the optimization algorithm is returned as a single concatenated string or 
        a flat array and needs to be separated into parts corresponding to each entity.

        Parameters:
            sol (str): The flat solution string or array from the optimization algorithm. It is assumed that
                    the solution contains binary representations for all entities concatenated together.
            evs (int): The number of evs
            Horizon (int): The optimization horizon

        Returns:
            list: A list of strings, where each string represents the binary solution for an individual entity.
        """
        solution = []
        for i in range(evs):
            solution.append(sol[i*2*Horizon:(i+1)*2*Horizon])
            
        return solution



    def get_qubomat(self, epsilon , de, C, Horizon, DT):
        """
        Combines various QUBO matrices into a single matrix for the entire optimization problem. This method 
        integrates different components of the problem, including NC terms, LV terms, 
        penalties, and QC, into a unified QUBO matrix that can be used for quantum optimization.

        Parameters:
            epsilon (list of float): Array of energy demand for each EV.
            de (list of int): Array of charging duration demand for each EV.
            C (float): Nework's Power capacit.
            Horizon (int): Optimization horizon, indicating the total duration over which the optimization is performed.
            DT (float): Time step size or duration.

        Returns:
            numpy.ndarray: A combined QUBO matrix for the optimization problem.
        """
        evs = len(epsilon)
        V=0.240
        # Q_NC  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
        # Q_LV  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
        # Q_P  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
        # Q_QC  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
        Q_NC  = np.zeros((evs*Horizon*2 ,evs*Horizon*2))
        Q_LV  = np.zeros((evs*Horizon*2 ,evs*Horizon*2))
        Q_P  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
        Q_QC  = np.zeros((evs*Horizon*2 ,evs*Horizon*2))
        
        for ev in range(evs):
            Q_NC[ev*Horizon*2:(ev+1)*Horizon*2,ev*Horizon*2:(ev+1)*Horizon*2] = self.get_qubomat_NC(epsilon[ev],de[ev],Horizon,V,DT)

        for evi in range(evs):
            for evj in range(evi,evs):
                Q_LV[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = self.get_qubomat_LV(evi,evj,de[evi],de[evj],Horizon)

                
        for evi in range(evs):
            for evj in range(evi,evs):
                Q_P[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = self.get_qubomat_P1(evi,evj,de[evi],de[evj],Horizon, V, C)

        Q_P[evs*Horizon*2: evs*Horizon*2 + Horizon*4, evs*Horizon*2 : evs*Horizon*2 + Horizon*4] = self.get_qubomat_P2(V, C, Horizon)
    
        for ev in range(evs):
            Q_P[ev*Horizon*2:(ev+1)*Horizon*2, evs*Horizon*2 : evs*Horizon*2 + Horizon*4] = self.get_qubomat_P3(de[ev], V,Horizon)

        Q_QC[0:evs*Horizon*2,0:evs*Horizon*2] = self.get_qubomat_QC(Horizon,evs)
            
        Q = Q_NC#500*Q_NC  + 1*Q_LV + 1*Q_QC# +100*(30/C)*Q_P

        return Q

    def optimize(self, n_measurements = 10000,number_of_experiments = 3,maxiter=200):

        self.inst.optimize(n_measurements = n_measurements,number_of_experiments = number_of_experiments,maxiter=maxiter)


    def get_sched(self):
        solution = self.inst.show_solution()

        solution = self.reshape_solution(solution,self.evs,self.Horizon)
        return self.ret_schedule(solution,self.evs,self.Horizon,self.de)
    
    def get_optimal_cost(self):
        return self.inst.optimal_cost
    
    def plot_evolution(self,normalization = [],label = "",color = 'C0'):
        self.inst.plot_evolution(normalization, label = label,color=color)

    def get_solution_distribution(self, normalization = [], label ="", marker= '*', solutions=10, shots=10000):
        self.inst.get_solution_distribution(normalization, label, marker, solutions,shots)

    def set_attributes(self, epsilon , de ,C, algorithm = "CompressedVQE"):
        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C
        Q = self.get_qubomat(self.epsilon , self.de, self.C, self.Horizon, self.DT)

        if algorithm == "CompressedVQE":
            inst = CompressedVQE(Q,6,na=2)
            #inst.optimize(n_measurements = n_measurements,number_of_experiments = number_of_experiments,maxiter=maxiter)
        else:
            inst = VQE(Q,6)
            #inst.optimize(n_measurements = n_measurements,number_of_experiments = number_of_experiments,maxiter=maxiter)

        self.inst = inst

    def add_noise(self,t1 = 50e3, t2 = 70e3, prob_1_qubit = 0.001 ,prob_2_qubit = 0.01, p1_0 = 0.2, p0_1 = 0.3):
        self.inst.add_noise(t1 = t1, t2 = t2, prob_1_qubit = prob_1_qubit ,prob_2_qubit = prob_2_qubit, p1_0 = p1_0, p0_1 = p0_1)
        
        

    #print(get_qubomat_QC(2,2))


from optimization_algorithm import *
from postprocessing import *
import numpy as np
from matplotlib import pyplot as plt
a=[2,2,2,2,2,2,2,2,2,2]
S=6

# Q = [
#         [-1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, -1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, 2.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
#         ]
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
Q = get_qubomat(a,S)
vqe_instance = CompressedVQE(Q,layers =1,na =1)
vqe_result = vqe_instance.optimize(maxiter=1000, shots=10000, experiments=5)
vqe_instance.plot_evolution()
vqe_instance.phow_solution()
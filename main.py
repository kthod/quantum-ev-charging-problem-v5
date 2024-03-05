from EV_Charging_problem import *
from postprocessing import *
import numpy as np
from matplotlib import pyplot as plt
a=[2,2,2,2]
S=6

Q = [
        [-1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        ]
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


#Q = get_qubomat(a,S)
epsilon = np.array([50000,50000,50000,50000,50000,50000,10000,10000])
d = np.array([6,6,6,6,6,6,1,1])
allowable_set = np.array([8,16,32,48,64])
vqe_instance3 = CompressedVQE3(Q,layers =2,evs = 8,horizon = 8,timestep=1,epsilon=epsilon, d = d)
vqe_result3 = vqe_instance3.optimize(maxiter=500, shots=10000, experiments=1)
print(np.sum(240*vqe_result3,axis = 1))
roud_res = diff_based_reallocation(vqe_result3,allowable_set,epsilon,d,240,1)
print("===========================================================")
print(roud_res)
print(np.sum(240*roud_res,axis = 1))
print(np.sum(240*vqe_result3,axis = 1))
supply = np.sum(240*roud_res,axis = 1)

diff = supply - epsilon
diff[diff>0] = 0

supply = epsilon + diff

demand_met = sum(supply)/sum(epsilon)*100
print(demand_met)

trl_vec = [1000]
demand_met_vec = []
demand_met_vec2 = []
demand_met_vec3 = []
for trl in trl_vec:
    vqe_instance = CompressedVQE(Q,layers =2,evs = 8,horizon = 8,timestep=1,total_rate_limit=trl,epsilon=epsilon, d = d)
    vqe_result = vqe_instance.optimize(maxiter=500, shots=10000, experiments=5)
    vqe_instance.plot_evolution()
    #print(vqe_result)
    
    roud_res = diff_based_reallocation(vqe_result,allowable_set,epsilon,d,240,1)
    #print(np.sum(240*vqe_result,axis = 1))
    print(roud_res)
    print(np.sum(240*roud_res,axis = 1))
    print(vqe_instance)

    supply = np.sum(240*vqe_result,axis = 1)

    diff = supply - epsilon
    diff[diff>0] = 0

    supply = epsilon + diff

    demand_met = sum(supply)/sum(epsilon)*100
    demand_met_vec.append(demand_met)
    vqe_instance2 = CompressedVQE2(Q,layers =2,evs = 8,horizon = 8,timestep=1,total_rate_limit=trl,epsilon=epsilon, d = d)
    vqe_result2 = vqe_instance2.optimize(maxiter=500, shots=10000, experiments=5)
    vqe_instance2.plot_evolution()
    roud_res = diff_based_reallocation(vqe_result2,allowable_set,epsilon,d,240,1)
    print("===========================================================")
    print(roud_res)
    print(np.sum(240*roud_res,axis = 1))

    supply = np.sum(240*vqe_result2,axis = 1)

    diff = supply - epsilon
    diff[diff>0] = 0

    supply = epsilon + diff

    demand_met = sum(supply)/sum(epsilon)*100
    demand_met_vec2.append(demand_met)

    vqe_instance3 = CompressedVQE3(Q,layers =2,evs = 8,horizon = 8,timestep=1,total_rate_limit=trl,epsilon=epsilon, d = d)
    vqe_result3 = vqe_instance3.optimize(maxiter=500, shots=10000, experiments=5)
    vqe_instance3.plot_evolution()
    roud_res = diff_based_reallocation(vqe_result3,allowable_set,epsilon,d,240,1)
    print("===========================================================")
    print(roud_res)
    print(np.sum(240*roud_res,axis = 1))

    supply = np.sum(240*vqe_result3,axis = 1)

    diff = supply - epsilon
    diff[diff>0] = 0

    supply = epsilon + diff

    demand_met = sum(supply)/sum(epsilon)*100
    demand_met_vec3.append(demand_met)


plt.plot([element * 240 / 1000 for element in trl_vec],demand_met_vec,label="method 2")
plt.plot([element * 240 / 1000 for element in trl_vec],demand_met_vec2,label="method 1")
plt.plot([element * 240 / 1000 for element in trl_vec],demand_met_vec3,label="method 3")
plt.xlabel("power capacity (kW)")
plt.ylabel("demand met (%)")
plt.legend()
plt.show()
#   def non_completion(rates):
#                 power = voltage*rates
#                 epsilon = self.espilon
#                 d = self.d
#                 sum = 0

#                 for i in range(self.num_of_EV):
#                     sum+= np.abs(np.sum(power[i,:d[i]]) -epsilon[i])

#                 return sum
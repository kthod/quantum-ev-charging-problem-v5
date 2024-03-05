from quantum_MPC import *
from classical_algorithms import *
import numpy as np
from matplotlib import pyplot as plt

def simulation (e,d, a, energy_limit,alg,timespan):
    departure = a +d
    total_number_of_EVs = len(e)
    Daily_schedule = np.zeros((total_number_of_EVs, timespan//DT))
    epsilon = e.copy()
    de = d.copy()
    for t in range(timespan//DT):
        temp_e = []
        temp_d = []
        active_EVs = []
        for ev in range(total_number_of_EVs):
            if a[ev] <= t < departure[ev]:
                active_EVs.append(ev)
                temp_e.append(epsilon[ev])
                temp_d.append(de[ev])

        alg.set_attributes(temp_e,temp_d,energy_limit)
        alg.optimize()
        optimized_schedules = alg.get_sched()
        for i in range(len(active_EVs)):
            epsilon[active_EVs[i]]-=240*DT*optimized_schedules[i,0]/1000
            de[active_EVs[i]]-=1
            Daily_schedule[active_EVs[i],t] = optimized_schedules[i,0]
    
    return Daily_schedule


# departure = a +de
def demand_met_comparison(epsilon,de,T,DT):
    energy_limits = [3*16*240/1000,4*16*240/1000,5*16*240/1000,6*16*240/1000,7*16*240/1000,8*16*240/1000,9*16*240/1000,10*16*240/1000,11*16*240/1000,12*16*240/1000,13*16*240/1000,14*16*240/1000,15*16*240/1000,16*16*240/1000,17*16*240/1000,18*16*240/1000,19*16*240/1000,20*16*240/1000,21*16*240/1000]

    j = 0
    # markers = ["s","o","*","^"]
    exp = 15
    demand_met_mat = np.zeros((exp,len(energy_limits)))
    alg = Quantum_MPC(epsilon,de,1,T,DT)
    for i in range(exp):  
        
        
        for j in range(len(energy_limits)):
            print("==========================================")
            print(energy_limits[j])
            print("==========================================")
            
            
            #Daily_schedule = simulation(epsilon,de,a,energy_limit,algorithms[i])
            alg.set_attributes(epsilon,de,energy_limits[j])
            alg.optimize()
            optimized_schedules = alg.get_sched()
            print(optimized_schedules)
            # epsilon = np.array([26880,26880,26880,7680])/1000
            # de = np.array([3,3,3,1])

            supply = np.sum(240*DT*optimized_schedules/1000,axis = 1)

            diff = supply - epsilon
            diff[diff>0] = 0

            supply = epsilon + diff

            demand_met_mat[i,j]=(sum(supply)/sum(epsilon)*100)

    upper_bound_min = [np.max(demand_met_mat[:, i]) for i in range(len(energy_limits))]
    lower_bound_min = [np.min(demand_met_mat[:, i]) for i in range(len(energy_limits))]
    mean_min = [np.mean(demand_met_mat[:, i]) for i in range(len(energy_limits))]



    plt.fill_between(energy_limits, upper_bound_min, lower_bound_min, alpha=0.4)
    plt.plot(energy_limits, mean_min, marker="s", linestyle='--', label="Quantum MPC")
    

    name = ["First-Come First-Serve","Earliest Deadline First","Least Laxity First"]
    algorithms = [FCFS(epsilon,de,1,T,DT),EDF(epsilon,de,1,DT),LLF(epsilon,de,1,T,DT)]
    markers = ["o","*","^"]
    for i in range(len(algorithms)):  
        demand_met = []
        alg = algorithms[i]
        for energy_limit in energy_limits:
            print("==========================================")
            print(energy_limit)
            print("==========================================")
            
            
            #Daily_schedule = simulation(epsilon,de,a,energy_limit,algorithms[i])
            alg.set_attributes(epsilon,de,energy_limit)
            alg.optimize()
            optimized_schedules = alg.get_sched()
            print(optimized_schedules)
            epsilon = np.array([26880,26880,26880,7680])/1000
            de = np.array([3,3,3,1])

            supply = np.sum(240*DT*optimized_schedules/1000,axis = 1)

            diff = supply - epsilon
            diff[diff>0] = 0

            supply = epsilon + diff

            demand_met.append(sum(supply)/sum(epsilon)*100)

        plt.plot(energy_limits, demand_met, marker = markers[i], label=name[i])


    plt.title("Demand met comparisson")
    plt.xlabel("Network's Power Capacity")
    plt.ylabel("Demand met (%)")
    plt.legend()
    plt.show()


def energy_consumption(epsilon,de,a,T,DT):
    energy_limit = 50*16*240/1000
    exp=10
    timespan = 24
    alg = Quantum_MPC(epsilon,de,1,T,DT)

    energy_mat = np.zeros((exp,timespan//DT))
    for i in range(exp):  
        
        

        Daily_schedule = simulation(epsilon,de,a,energy_limit,alg,timespan)



        energy_mat[i,:] = np.sum(240*Daily_schedule/1000,axis = 0)


        

    upper_bound_min = [np.max(energy_mat[:, i]) for i in range(timespan//DT)]
    lower_bound_min = [np.min(energy_mat[:, i]) for i in range(timespan//DT)]
    mean_min = [np.mean(energy_mat[:, i]) for i in range(timespan//DT)]



    plt.fill_between(range(timespan//DT), upper_bound_min, lower_bound_min, alpha=0.4)
    plt.plot(range(timespan//DT), mean_min, marker="s", linestyle='--', label="Quantum MPC")

    name = ["First-Come First-Serve","Earliest Deadline First","Least Laxity First"]
    algorithms = [FCFS(epsilon,de,energy_limit,T,DT),EDF(epsilon,de,energy_limit,T,DT),LLF(epsilon,de,energy_limit,T,DT)]
    markers = ["o","*","^"]
    for i in range(len(algorithms)):  
        
            
            
        Daily_schedule = simulation(epsilon,de,a,energy_limit,algorithms[i],timespan)


        epsilon = 2*np.array(10*[26880,26880,26880,7680,23040,23040,7680,7680,7680,7680])/1000
        de = 2*np.array(10*[3,3,3,1,3,3,1,1,1,1])

        energy = np.sum(240*Daily_schedule/1000,axis = 0)


        plt.plot(range(timespan//DT), energy, marker = markers[i], label=name[i])


    plt.title("Energy Consumption comparisson")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Total Power (kW)")
    plt.legend()
    plt.show()

    



DT = 1
T=4

epsilon = 2*np.array(10*[26880,26880,26880,7680,23040,23040,7680,7680,7680,7680])/1000
de = 2*np.array(10*[3,3,3,1,3,3,1,1,1,1])
a = 1*np.array(5*[0,2,4,6,7,8,10,12,14,16,1,3,7,9,11,15,17,19,21,22])
#demand_met_comparison(epsilon,de,T,DT)
energy_consumption(epsilon,de,a,6,1)
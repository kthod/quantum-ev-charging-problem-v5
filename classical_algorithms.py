import numpy as np


class FCFS():
    def __init__(self,epsilon , de, C, Horizon, DT):

        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C
        self.Horizon = Horizon
        self.DT = DT

        self.sched = None
        
    def optimize(self):
        voltage = 240  # Constant voltage
        currents = [0, 16, 32, 48]  # Available currents
        
        
        e = self.epsilon
        d = self.de
        energy_limit = self.C
        T = self.Horizon
        DT = self.DT
        evs_charging_schedules = np.zeros((len(e),T))
        energy_limits = np.array(T*[energy_limit])
        def ev_charging_plan(ev):
            #print(energy_limits)
            remaining_energy_demand = e[ev]
            remaining_duration = d[ev]
            charging_schedule = []
            
            for timestep in range(T):
                # Determine the best current for this hour
                best_current = 0
                min_energy_diff = float('inf')
                if timestep < remaining_duration:
                    for current in currents:
                        energy_this_timestep = current * voltage * DT / 1000  # kWh
                        # Predict remaining energy demand if this current is used
                        predicted_remaining = remaining_energy_demand - energy_this_timestep
                        
                        # Choose the current that gets closest to fulfilling the demand without overshooting
                        if 0 <= predicted_remaining < min_energy_diff and energy_this_timestep <= energy_limits[timestep]:
                            best_current = current
                            min_energy_diff = predicted_remaining
                
                # Update charging schedule and remaining energy demand
                charging_schedule.append(best_current)
                remaining_energy_demand -= best_current * voltage * DT / 1000
            
            # Ensure the EV's energy demand is met or adjusted in the last hour if necessary
            # if remaining_energy_demand > 0 and charging_schedule:
            #     # Adjust the last hour to meet the exact demand if not met
            #     additional_energy_needed = remaining_energy_demand * 1000 / voltage  # Convert back to Amps
            #     charging_schedule[d[ev]-1] = min(currents, key=lambda x: abs(x - additional_energy_needed))
            
            return charging_schedule
        
        for ev in range(len(e)):
            evs_charging_schedules[ev ,:] = ev_charging_plan(ev)
            energy_limits = energy_limits - voltage*DT*evs_charging_schedules[ev, :]/1000

        
        self.sched = evs_charging_schedules

    def get_sched(self):
        return self.sched
    
    def set_attributes(self, epsilon , de ,C):
        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C



class LLF():
    def __init__(self,epsilon , de, C, Horizon, DT):

        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C
        self.Horizon = Horizon
        self.DT = DT

        self.sched = None

    def optimize(self):
        e = self.epsilon.copy()
        voltage = 240  # Constant voltage
        currents = [0, 16, 32, 48]  # Available currents
        
        
       
        d = self.de
        energy_limit = self.C
        T = self.Horizon
        DT = self.DT
        evs_charging_schedules = np.zeros((len(e), T))
        energy_limits = np.array(T * [energy_limit])
        def calculate_laxity(remaining_energy_demand, remaining_duration, max_current):
            max_energy_per_hour = max_current * voltage * DT / 1000  # kWh
            min_hours_required = np.ceil(remaining_energy_demand / max_energy_per_hour)
            laxity = remaining_duration - min_hours_required
            return laxity

        for timestep in range(T):
            # Calculate laxity for each EV
            laxities = []
            for ev in range(len(e)):
                laxity = calculate_laxity(e[ev], d[ev] - timestep, currents[-1])
                laxities.append((laxity, ev))

            # Sort EVs by laxity
            sorted_evs_by_laxity = sorted(laxities, key=lambda x: x[0])
            
            for laxity, ev in sorted_evs_by_laxity:
                remaining_energy_demand = e[ev]
                remaining_duration = d[ev] - timestep
                best_current = 0
                min_energy_diff = float('inf')

                if remaining_duration > 0:
                    for current in currents:
                        energy_this_timestep = current * voltage * DT / 1000  # kWh
                        predicted_remaining = remaining_energy_demand - energy_this_timestep
                        
                        if 0 <= predicted_remaining < min_energy_diff and energy_this_timestep <= energy_limits[timestep]:
                            best_current = current
                            min_energy_diff = predicted_remaining

                # Update charging schedule and remaining energy demand
                evs_charging_schedules[ev, timestep] = best_current
                e[ev] -= best_current * voltage * DT / 1000

            # Update energy limits
                energy_limits = energy_limits - voltage*DT*evs_charging_schedules[ev, :]/1000

        self.sched = evs_charging_schedules

    def get_sched(self):
        return self.sched
    
    def set_attributes(self, epsilon , de ,C):
        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C

    # Example EVs


# Calculate optimized charging schedules using LLF

class EDF():
    def __init__(self,epsilon , de, C, Horizon, DT):

        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C
        self.Horizon = Horizon
        self.DT = DT

        self.sched = None

    def optimize(self):
        e= self.epsilon.copy()
        voltage = 240  # Constant voltage
        currents = [0, 16, 32, 48]  # Available currents
        
       
        d = self.de
        energy_limit = self.C
        T = self.Horizon
        DT = self.DT
        evs_charging_schedules = np.zeros((len(e), T))
        energy_limits = np.array(T * [energy_limit])
        # Sort EVs by deadline
        ev_deadlines = sorted(range(len(d)), key=lambda x: d[x])
        for timestep in range(T):
            for ev in ev_deadlines:
                remaining_energy_demand = e[ev]
                remaining_duration = d[ev] - timestep
                best_current = 0
                min_energy_diff = float('inf')

                if remaining_duration > 0:
                    for current in currents:
                        energy_this_timestep = current * voltage * DT / 1000  # kWh
                        predicted_remaining = remaining_energy_demand - energy_this_timestep

                        if 0 <= predicted_remaining < min_energy_diff and energy_this_timestep <= energy_limits[timestep]:
                            best_current = current
                            min_energy_diff = predicted_remaining

                # Update charging schedule and remaining energy demand
                evs_charging_schedules[ev, timestep] = best_current
                e[ev] -= best_current * voltage * DT / 1000

            # Update energy limits
                energy_limits = energy_limits - voltage*DT*evs_charging_schedules[ev, :]/1000

        self.sched = evs_charging_schedules

    def get_sched(self):
        return self.sched
    
    def set_attributes(self, epsilon , de ,C):
        self.epsilon = epsilon
        self.evs = len(epsilon)
        self.de = de
        self.C = C

# Example EVs


# Calculate optimized charging schedules using EDF



# # Example EVs
# e = np.array(1*[26880,26880,26880,7680,23040,23040,7680,7680])/1000
# d = 1*np.array(1*[3,3,3,1,3,3,1,1])
# energy_limit = 50
# T = 4
# DT = 1

# # Calculate optimized charging schedules
# optimized_schedules = optimize_charging_schedule(e , d, energy_limit, T, DT)

# print(optimized_schedules)


# optimized_schedules_llf = optimize_charging_schedule_llf(e, d, energy_limit, T, DT)
# print(optimized_schedules_llf)

# optimized_schedules_edf = optimize_charging_schedule_edf(e, d, energy_limit, T, DT)
# print(optimized_schedules_edf)

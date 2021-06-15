import numpy as np
import networkx as nx
import random as rand
import statistics
import math
import matplotlib.pyplot as plt

def initial_solution(ETC, ready_times):
    #Input: ETC = m x n matrix, m is the number of machines, n is the number of jobs
    #ready_times = vector containing the ready time of each machine
    #Output: Min-Min solution of the problem

    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]
    completion = ETC.copy()
    for i in range(m):
        for j in range(n):
            completion[i][j] += ready_times[i]

    initial_solution = [0 for j in range(n)]
    place_holder = np.max(completion) + 1

    for j in range(n):
        minimum = np.min(completion)
        minimum_index = np.argwhere(completion == minimum)[0].tolist()
        initial_solution[minimum_index[1]] = minimum_index[0]
        completion[:,minimum_index[1]] = place_holder
    
    return initial_solution

def makespan(ETC, schedule):
    #Input: ETC = m x n matrix, m is the number of machines, n is the number of jobs
    #schedule = vector containing which job is allocated to which machine
    #Output: Makespan of the schedule
    n = np.shape(ETC)[1]
    list_of_times = []
    for i in range(n):
        list_of_times.append(ETC[schedule[i]][i])
    return max(list_of_times)

def flowtime(ETC, schedule):
    #Input: ETC = m x n matrix, m is the number of machines, n is the number of jobs
    #schedule = vector containing which job is allocated to which machine
    #Output: Flowtime of the schedule
    n = np.shape(ETC)[1]
    list_of_times = []
    for i in range(n):
        list_of_times.append(ETC[schedule[i]][i])
    
    return sum(list_of_times)

def roulette(initial_list):
    """
        Input: list of positive numbers.
        Output: A 2-tuple containing a random element from the Python list other than the maximum and 
        its index.
    """
    # remove the minimum of initial_list
    second_list = initial_list.copy()
    minimum = min(second_list)
    second_list.remove(minimum)

    # assign a probability distribution to all non-minimum elements in initial_list
    # the probability of getting x should be inversely proportional to x
    reciprocals = np.array([1/x for x in second_list])
    probabilities = reciprocals/np.sum(reciprocals)

    # draw x from the roulette distribution and find the index of x
    random_number = np.random.choice(second_list, p=probabilities) 
    number_index = initial_list.index(random_number)

    return random_number, number_index

def generate_random_jobs(mj = 10, a = 0, b = 10):
    
    #Input: mj = maximum number of jobs or machines. a = minimum time for a job to finish, b = maximum time for a job to finish
    #Output: Randomly generated ETC matrix and ready times vector
    
    m = rand.choice(range(2, mj+1)) # number of machines
    n = rand.choice(range(2, mj+1)) # number of jobs
    ready_times = []
    ETC = np.zeros((m, n))
    
    for i in range(m):
        tm = rand.randint(a, b)
        ready_times.append(tm)
    
    for i in range(m):
        for j in range(n):
            tm = rand.randint(a, b)
            ETC[i][j] = tm
    
    return ETC, ready_times

def completion_time(ETC, ready_times, solution):

    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]
    completion_list = ready_times.copy()
    for i in range(m):
        for j in range(n):
            if solution[j] == i:
                completion_list[i] += ETC [i][j]
    return max(completion_list)

def diversification_search(ETC, ready_times, diversification_iterations = 5000, swaps = 10, transfers = 10, L = 0.5, p = 0.8):
    #Input: ETC = m x n matrix, m is the number of machines, n is the number of jobs
    #       ready_times = m-vector containing the ready time of each machine
    #       diversification_iterations = number of iterations for which we apply the diversification procedures; first iterations
    #       swaps = number of swaps per iteration
    #       transfers = number of transfers per iteration
    #       L = parameter for determining fitness
    #       p = probability of picking a good solution during diversification
    
    #Output: Schedule obtained from diversification

    #Initializing variables    
    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]
    algo_solution = initial_solution(ETC, ready_times)
    best_solution = algo_solution

    """change parameters"""
    #number of job_change_list considered
    change_list_length = 10 #can be changed
    
    #maximum number of times a job changes recently
    change_sum_limit = math.floor(0.4*change_list_length) #change the coefficient only

    frequency_list_length = math.floor(2/5*m*n) #try m*n or (m+n)

    tabu_list_length = 2 * math.floor(2/5*m*n) #only change what's inside the fllor function, try m*n or (m+n) (but keep the same as what you choose in frequency_list_length)

    # Choose penalty base parameter, only change things inside the fllor function, should be a function of frequency_list_length
    penalty_limit = math.floor(3/4* frequency_list_length)

    #Dont change first two lines, only change the coefficient of penalty_parameter
    i_s = initial_solution(ETC, ready_times)
    initial_fitness = L*makespan(ETC, i_s) + (1-L)*flowtime(ETC, i_s)/m
    penalty_parameter = initial_fitness/(20 * frequency_list_length)

    """end of changing parameters"""
    
    G = nx.Graph() #Graph of the solution; we use a frequency matrix the same way as we did for the TSP
                   #However, this time we have an m+n matrix. The upper left m x m submatrix and the lower right ones will be 0
                   #The matrices we need are the remaining bits, namely the first m rows intersected with the last n columns
    edges = []
    
    for job in range(n):
        edges.append([job + m, algo_solution[job]])
        
    G.add_edges_from(edges)
    A = nx.to_numpy_array(G, nodelist = range(m+n))
    frequency_list = [] 
    frequency_list.append(A)
    change_list = [] #The change list; will be used to create freeze lists.
                     #Number of times each job was changed
        
    fitness = L*makespan(ETC, algo_solution) + (1-L)*flowtime(ETC, algo_solution)/m
    best_fitness = fitness
    completion_list = ready_times.copy()
    
    for i in range(m):
        for j in range(n):
            if algo_solution[j] == i:
                completion_list[i] += ETC [i][j]
    
    completion_time = max(completion_list)
    tabu_swap = []
    tabu_transfer = []
    freeze_list = []

    #Starting the iterations
    for t in range(diversification_iterations):

        #As before, we create dummy variables specific for this iteration.
        iteration_allocations=[]
        iteration_tabu_swap=[]
        iteration_tabu_transfer=[]
        iteration_matrices=[]
        iteration_fitness=[]
        iteration_fitness_swap=[]
        iteration_fitness_transfer=[]
        iteration_change_list=[]
        iteration_freeze_list=[]
        iteration_completion_time=[]
        iteration_change=[]
        
        #Computing the busy machines and the less-busy ones; will be useful when we do the transfer movements.
        adj_matrix = frequency_list[-1]
        jobs_per_machine = [0 for i in range(m)]
        for index in range(m):
            for j in range(n):
                jobs_per_machine[index] += adj_matrix[index][j+m]
        
        #Doing the swaps
        for s in range(swaps):
            permutation = rand.sample(range(n), 2)
            job_1 = permutation[0]
            job_2 = permutation[1]
            swap_change = [0 for i in range(n)] #Creating a vector containing 1 if a job changed machine and 0 otherwise. 
                                       #This vector will be added to the change sum.
            
            for i in range(n):
                if i == job_1 or i == job_2:
                    swap_change[i] = 1

            swap_change = np.array(swap_change)

            #Creating the new solution and computing its graph and adjacency matrix.
            new_solution = algo_solution.copy()
            new_solution[job_1] = algo_solution[job_2]
            new_solution[job_2] = algo_solution[job_1]
            G_new = nx.Graph()
            edges_new = []

            for job in range(n):
                edges_new.append([job + m, new_solution[job]])

            G_new.add_edges_from(edges_new)
            A_new = nx.to_numpy_array(G_new, nodelist = range(m+n))
            
            #Checking if the frequency list is too small; if it isn't, remove the first element.
            new_frequency_list = frequency_list.copy()
            new_frequency_list.append(A_new)

            if len(new_frequency_list) > frequency_list_length:
                del new_frequency_list[0]
                
            matrix_sum = sum([i for i in new_frequency_list]) #Exactly as with the TSP.
            
            #Checking if the change list is too small; if it isn't, remove the first element.
            new_change_list = change_list.copy()
            new_change_list.append(swap_change)

            if len(new_change_list) > change_list_length:
                del new_change_list[0]
            
            change_sum = sum([i for i in new_change_list]) #Similar to the TSP
            
            #Creating a freeze list.
            new_freeze_list = []
            
            for i in range(n):
                if change_sum[i] > change_sum_limit:  #This value may be changed, maybe turn it into a function of m and n?
                    new_freeze_list.append(i)
            
            #Computing the completion time and fitness, taking the frequency penalty into account.
            
            job_time = []
            
            for i in range(n):
                if matrix_sum[new_solution[i]][i+m] > penalty_limit:
                    job_time.append(ETC[new_solution[i]][i] + penalty_parameter *matrix_sum[new_solution[i]][i+m]) #Penalty may be changed, I'd suggest making it a function of the biggest completion time
                else:
                    job_time.append(ETC[new_solution[i]][i])
            
            new_makespan = max(job_time)
            new_flowtime = sum(job_time)
            new_fitness = L*new_makespan + (1-L)*new_flowtime/m
            new_completion_list = ready_times.copy()  #The list of completion times; we find the maximum value of this list and then use it for the aspiration condition.
            
            for i in range(m):
                for j in range(n):
                    if new_solution[j] == i:
                        new_completion_list[i] += job_time[j]
                        
            new_completion_time = max(new_completion_list)
            
            #Checking the conditions: first the aspiration ones, then the typical ones; exactly like TSP
            
            if ((new_fitness < fitness or new_completion_time < completion_time) and ([job_1, job_2] in tabu_swap) and tabu_swap.index([job_1, job_2]) < 2) \
                or ([job_1, job_2] not in tabu_swap):
                if (job_1 not in freeze_list) and (job_2 not in freeze_list):
                    iteration_tabu_swap.append([[job_1, job_2], [job_2, job_1]])
                    iteration_change_list.append(new_change_list)
                    iteration_freeze_list.append(new_freeze_list)
                    iteration_matrices.append(A_new)
                    iteration_allocations.append(new_solution)
                    iteration_completion_time.append(new_completion_time)
                    iteration_fitness.append(new_fitness)
                    iteration_change.append(swap_change)
                    iteration_fitness_swap.append(new_fitness)
   
        #Computing which machines are busy and which are not; if a machine has more than average jobs, we consider it busy; if not, it's free.
        average = statistics.mean(jobs_per_machine)
        busy_machines = []
        free_machines = []
            
        for machine in range(m):
            if jobs_per_machine[machine] >= average:
                busy_machines.append(machine)
            else:
                free_machines.append(machine)

        #Starting the transfers and randomly picking. 
        for trs in range(transfers):

            #All the machines may have the same number of jobs; checking for that.
            if free_machines:
                machine_1 = rand.choice(busy_machines) #Picking the machines randomly.
                machine_2 = rand.choice(free_machines)
            
            else:
                machines = rand.sample(busy_machines, 2)
                machine_1 = machines[0]
                machine_2 = machines[1]
            
            machine_1_jobs = [] #Creating the list of jobs for the first machine, so that we can randomly select one from those.

            for i in range(n):
                if algo_solution[i] == machine_1:
                    machine_1_jobs.append(i)

            job = rand.choice(machine_1_jobs) #Common bug: for some reason, this one is empty; FIXED
            new_solution = algo_solution.copy() #Rest of the algorithm is very similar to the swap one.
            new_solution[job] = machine_2
            swap_change = [0 for i in range(n)]

            for i in range(n):
                if i == job:
                    swap_change[i] = 1
            
            swap_change = np.array(swap_change)
    
            G_new = nx.Graph()
            edges_new = []
                
            for job in range(n):
                edges_new.append([job + m, new_solution[job]])
                    
            G_new.add_edges_from(edges_new)
            A_new = nx.to_numpy_array(G_new, nodelist = range(m+n))
                
            #Checking if the frequency list is too small; if it isn't, remove the first element
            new_frequency_list = frequency_list.copy()
            new_frequency_list.append(A_new)

            if len(new_frequency_list) > frequency_list_length:
                del new_frequency_list[0]
                
            matrix_sum = sum([i for i in new_frequency_list])
            
            #Checking if the change list is too small; if it isn't, remove the first element.
            new_change_list = change_list.copy()
            new_change_list.append(swap_change)
            
            if len(new_change_list) > change_list_length:
                del new_change_list[0]
            
            change_sum = sum([i for i in new_change_list])
            
            #Creating a freeze list.
            new_freeze_list=[]
            for i in range(n):
                if change_sum[i] > change_sum_limit:
                    new_freeze_list.append(i)
            
            #Computing the completion time and fitness, taking the frequency penalty into account.
            job_time = []
            
            for i in range(n):
                if matrix_sum[new_solution[i]][i+m] > penalty_limit:
                    job_time.append(ETC[new_solution[i]][i] + penalty_parameter*matrix_sum[new_solution[i]][i+m])
                else:
                    job_time.append(ETC[new_solution[i]][i])
            
            new_makespan = max(job_time)
            new_flowtime = sum(job_time)
            new_fitness = L*new_makespan + (1-L)*new_flowtime/m             

            new_completion_list = ready_times.copy()
            for i in range(m):
                for j in range(n):
                    if new_solution[j] == i:
                        new_completion_list[i] += job_time[j]    
            new_completion_time = max(new_completion_list)
            
            #Checking the conditions; first the aspiration ones, then the classical ones.
            #The tabu list here is different, that's why we used two separate tabu lists; the swap one and the transfer one
            #The transfer one contains the job that was changed and the machines that were part of the transfer
            
            if ((new_fitness < fitness or new_completion_time < completion_time) and ([job, machine_1, machine_2] in tabu_transfer) and tabu_transfer.index([job, machine_1, machine_2]) < 2) \
                or ([job, machine_1, machine_2] not in tabu_transfer):
                if (job not in freeze_list):
                    iteration_tabu_transfer.append([[job, machine_1, machine_2], [job, machine_2, machine_1]]) 
                    iteration_change_list.append(new_change_list)
                    iteration_freeze_list.append(new_freeze_list)
                    iteration_matrices.append(A_new)
                    iteration_allocations.append(new_solution)
                    iteration_completion_time.append(new_completion_time)
                    iteration_fitness.append(new_fitness)
                    iteration_change.append(swap_change)
                    iteration_fitness_transfer.append(new_fitness)
            
        #Applying the roulette/probability  method as in the TSP algorithm        
        coin_toss = rand.uniform(0,1)
        if len(iteration_fitness) > 1:
            if coin_toss < p:
                fitness = min(iteration_fitness)
                index = iteration_fitness.index(fitness)
            else:
                roul = roulette(iteration_fitness)
                index = roul[1]
                fitness = iteration_fitness[index]
            algo_solution = iteration_allocations[index]
            change_vector = iteration_change[index]
            change_list = iteration_change_list[index]
            freeze_list = iteration_freeze_list[index]

            #Checking if the appended value is a swap or a transfer; for a transfer, we make only one change in the job list, so 
            #the sum of change_vector is 1; for a swap, it's 2.

            if sum(change_vector) == 1:
                fitness_transfer_index = iteration_fitness_transfer.index(fitness)
                tabu_transfer.append(iteration_tabu_transfer[fitness_transfer_index][0])
                tabu_transfer.append(iteration_tabu_transfer[fitness_transfer_index][1])
                if len(tabu_transfer) > tabu_list_length:
                    del tabu_transfer[0:2]
            if sum(change_vector) == 2:
                fitness_swap_index = iteration_fitness_swap.index(fitness)
                tabu_swap.append(iteration_tabu_swap[fitness_swap_index][0])
                tabu_swap.append(iteration_tabu_swap[fitness_swap_index][1])
                if len(tabu_swap) > tabu_list_length:
                    del tabu_swap[0:2]
            frequency_list.append(iteration_matrices[index])
        
        if len(iteration_fitness) == 1:
            algo_solution = iteration_allocations[0]
            change_vector = iteration_change[0]
            change_list = iteration_change_list[0]
            freeze_list = iteration_freeze_list[0]
            if sum(change_vector) == 1:
                tabu_transfer.append(iteration_tabu_transfer[0][0])
                tabu_transfer.append(iteration_tabu_transfer[0][1])
                if len(tabu_transfer) > tabu_list_length:
                    del tabu_transfer[0:2]
            if sum(change_vector) == 2:
                tabu_swap.append(iteration_tabu_swap[0][0])
                tabu_swap.append(iteration_tabu_swap[0][1])
                if len(tabu_swap) > tabu_list_length:
                    del tabu_swap[0:2]
            frequency_list.append(iteration_matrices[0])
        
        #Computing the best fitness we found thus far
        
        #Computing the best fitness we found thus far
        true_fitness = L*makespan(ETC, algo_solution) + (1-L)*flowtime(ETC, algo_solution)/m
        if true_fitness < best_fitness:
            best_solution = algo_solution
            best_fitness = true_fitness
    
    return best_solution, best_fitness

def intensification_search(ETC, ready_times, solution, intensification_iterations = 5000, L = 0.5):
    
    #Input: ETC = m x n matrix, m is the number of machines, n is the number of jobs
    #       ready_times = vector containing the ready time for each machine
    #       intensification_iterations = number of iterations
    #       L = fitness constant
    
    #Output: best_solution = best solution found, best_fitness = fitness of the best solution
    
    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]
    algo_solution = solution
    best_solution = algo_solution
    fitness = L*makespan(ETC, algo_solution) + (1-L)*flowtime(ETC, algo_solution)/m
    best_fitness = fitness
    completion_list = ready_times.copy()

    """change parameters: just copy and paste what you define in diversification search function!!!"""

    tabu_list_length = 2 * math.floor(2/5*m*n) #only change what's inside the fllor function, try m*n or (m+n) (but keep the same as what you choose in frequency_list_length)

    """end of changing parameters"""
    
    for i in range(m):
        for j in range(n):
            if algo_solution[j] == i:
                completion_list[i] += ETC [i][j]
    
    completion_time = max(completion_list)
    change_vector = [0 for i in range(n)]
    tabu_swap = []
    tabu_transfer = []
    
    for t in range(intensification_iterations):
        #Doing all the possible swaps between two randomly selected machines with at least 1 job assigned
        
        iteration_allocations=[]
        iteration_tabu_swap=[]
        iteration_tabu_transfer=[]
        iteration_fitness=[]
        iteration_fitness_swap=[]
        iteration_fitness_transfer=[]
        iteration_completion_time=[]
        iteration_change_vectors=[]

        G_new = nx.Graph() 
        edges = []
    
        for job in range(n):
            edges.append([job + m, algo_solution[job]])
        
        G_new.add_edges_from(edges)
        A_new = nx.to_numpy_array(G_new, nodelist = range(m+n))

        adj_matrix = A_new
        jobs_per_machine = [0 for i in range(m)]
        for index in range(m):
            for j in range(n):
                jobs_per_machine[index] += adj_matrix[index][j+m]
        
        job_machines = [] #Creating a list of machines that are doing at least a job

        for i in range(m):
            if jobs_per_machine[i]:
                job_machines.append(i)

        if len(job_machines) > 1:
            machines = rand.sample(job_machines,2)
            machine_1 = machines[0]
            machine_2 = machines[1]
            machine_1_jobs=[]
            machine_2_jobs=[]
            
            for i in range(n):
                if algo_solution[i] == machine_1:
                    machine_1_jobs.append(i)
                if algo_solution[i] == machine_2:
                    machine_2_jobs.append(i)
            
            if len(machine_1_jobs) > len(machine_2_jobs):
                busy_machine_jobs = machine_1_jobs
                free_machine_jobs = machine_2_jobs
                busy_machine = machine_1
                free_machine = machine_2
            else:
                busy_machine_jobs = machine_2_jobs
                free_machine_jobs = machine_1_jobs
                busy_machine = machine_2
                free_machine = machine_1

            job = rand.choice(busy_machine_jobs)

            job_pair_list = [(busy_job, free_job) for busy_job in busy_machine_jobs for free_job in free_machine_jobs]

            for job_pair in job_pair_list:
                busy_job = job_pair[0]
                free_job = job_pair[1]
                new_solution = list(algo_solution)
                new_solution[busy_job] = free_machine
                new_solution[free_job] = busy_machine
                new_solution[job] = free_machine
                new_fitness = L * makespan(ETC, new_solution) + (1 - L) * flowtime(ETC, new_solution) / m
                new_change_vector = list(change_vector)
                new_change_vector[busy_job] = 1
                new_change_vector[free_job] = 1
                job_time = []

                for i in range(n): 
                    job_time.append(ETC[new_solution[i]][i])
            
                new_completion_list = list(range(m))

                for i in range(m):
                    new_completion_list[i] = ready_times[i]
                    for j in range(n):
                        if new_solution[j] == i:
                            new_completion_list[i] += job_time[j]
                
                new_completion_time = max(new_completion_list)
                
                if ((new_fitness < fitness or new_completion_time < completion_time) and [busy_job, free_job] in tabu_swap and tabu_swap.index([busy_job, free_job]) < 2)\
                    or ([busy_job, free_job] not in tabu_swap):
                    if ((new_fitness < fitness or new_completion_time < completion_time) and [job, busy_machine, free_machine] in tabu_transfer and tabu_transfer.index([job, busy_machine, free_machine]) < 2)\
                        or ([job, busy_machine, free_machine] not in tabu_transfer):
                        iteration_allocations.append(new_solution)
                        iteration_tabu_swap.append([[busy_job, free_job], [free_job, busy_job]])
                        iteration_tabu_transfer.append([[job, busy_machine, free_machine], [job, busy_machine, free_machine]])
                        iteration_fitness.append(new_fitness)
                        iteration_completion_time.append(new_completion_time)
                        iteration_fitness_swap.append(new_fitness)
                        iteration_fitness_transfer.append(new_fitness)
                        iteration_change_vectors.append(new_change_vector)

        if len(job_machines) == 1:
        
            machine_1 = job_machines[0]
            machine_list = list(range(m))
            del machine_list[machine_1]
            machine_2 = rand.choice(machine_list)
            job = rand.choice(range(n))
            new_solution = list(algo_solution)
            new_solution[job] = machine_2
            new_fitness = L * makespan(ETC, new_solution) + (1 - L) * flowtime(ETC, new_solution) / m
            new_change_vector = list(change_vector)
            new_change_vector[job]=1
            job_time = []

            for i in range(n): 
                job_time.append(ETC[new_solution[i]][i])
                
            new_completion_list = list(range(m))
            
            for i in range(m):
                new_completion_list[i] = ready_times[i]
                for j in range(n):
                    if new_solution[j] == i:
                        new_completion_list[i] += job_time[j]

            new_completion_time = max(new_completion_list)

            if ((new_fitness < fitness or new_completion_time < completion_time) and ([job, machine_1, machine_2] in tabu_transfer) and tabu_transfer.index([job, machine_1, machine_2]) < 2) \
                or ([job, machine_1, machine_2] not in tabu_transfer):
                iteration_allocations.append(new_solution)
                iteration_tabu_transfer.append([[job, machine_1, machine_2], [job, machine_2, machine_1]])
                iteration_fitness.append(new_fitness)
                iteration_completion_time.append(new_completion_time)
                iteration_fitness_transfer.append(new_fitness)
                iteration_change_vectors.append(new_change_vector)
    
        #We no longer use the roulette; we take the best solution
        if iteration_fitness:
            fitness = min(iteration_fitness)
            index = iteration_fitness.index(fitness)
            algo_solution = iteration_allocations[index]
            change_vector = iteration_change_vectors[index]
            tabu_transfer.append(iteration_tabu_transfer[0][0])
            tabu_transfer.append(iteration_tabu_transfer[0][1])
            if len(tabu_transfer) > tabu_list_length:
                del tabu_transfer[0:2]

            if sum(change_vector) == 2:
                tabu_swap.append(iteration_tabu_swap[index][0])
                tabu_swap.append(iteration_tabu_swap[index][1])
                if len(tabu_swap) > tabu_list_length:
                    del tabu_swap[0:2]
        
        true_fitness = L * makespan(ETC, algo_solution) + (1 - L) * flowtime(ETC, algo_solution)/m
        if true_fitness < best_fitness:
            best_solution = list(algo_solution)
            best_fitness = true_fitness
    
    return best_solution, best_fitness

def job_tabu_search(ETC, ready_times, diversification_iterations = 5000, intensification_iterations = 2000, swaps = 10, transfers = 10, L = 0.5, p = 0.8):
    #Input: ETC = m x n matrix, m is the number of machines, n is the number of jobs
    #       ready_times = m-vector containing the ready time of each machine
    #       diversification_iterations = number of iterations for which we apply the diversification procedures; first iterations
    #       intensification_iterations = number of iterations for which we apply the intensification procedures; last iterations
    #       swaps = number of swaps per iteration
    #       transfers = number of transfers per iteration
    #       L = parameter for determining fitness
    #       p = probability of picking a good solution during diversification
    
    #Output: Schedule obtained from tabu-searching
    
    #Initializing variables
    
    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]

    diversification_result = diversification_search(ETC, ready_times, diversification_iterations, swaps, transfers, L, p)
    solution = list(diversification_result[0])
    intensification_result = intensification_search(ETC, ready_times, solution, intensification_iterations, L)
    best_solution = intensification_result[0]
    best_fitness = intensification_result[1]
    completion = completion_time(ETC, ready_times, best_solution)
    
    return best_solution, best_fitness, completion

def working_time(ETC, ready_times, solution):

    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]
    full_working_time = []
    for i in range(m):
        starting_time = ready_times[i]
        working_time = [ready_times[i]]
        
        for j in range(n):
            if solution[j] == i:
                starting_time += ETC [i][j]
                working_time.append(starting_time)
                
        full_working_time.append(working_time)
        
    print(full_working_time)
    return full_working_time

def draw_job(ETC, ready_times, solution):
    
    m = np.shape(ETC)[0]
    n = np.shape(ETC)[1]
    
    # Declaring a figure "gnt"
    fig, gnt = plt.subplots()
 
    # Setting Y-axis limits
    gnt.set_ylim(-10, m*10)
 
    # Setting X-axis limits
    span = completion_time(ETC, ready_times, solution)
    gnt.set_xlim(0, span)
     
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('time')
    gnt.set_ylabel('machine')
 

    # Labelling tickes of y-axis
    gnt.set_yticks([i*10 for i in range(m)])
    gnt.set_yticklabels([f'{i}' for i in range(m)])
    
    full_work_schedule =  working_time(ETC, ready_times, solution)
    
    for j in range(m):
        gnt.broken_barh([(full_work_schedule[j][k], full_work_schedule[j][k+1] - 1) for k in range(len(full_work_schedule[j])-1)], (j*10-5, 9), facecolors ='white', edgecolor='black')

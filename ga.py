import os
import numpy as np
import pandas as pd
import time
import random
from neutrans import calculation, visualization

def fitness(w, x, y, z): # Fitness Function
    curr_keff = calculation(w, x, y, z)
    
    return curr_keff

start = time.time() #Start timer

# Initialize Variables
c = 0
prev_bestsolutions = np.empty((0, 5))
solutions = np.empty((0, 4), float)
pathcsv = 'inputs/genepool.csv'
exist = 0
mutation_rate = 0.16

#Generate Solutions
if os.path.isfile(pathcsv) == 1: # Check whether the file of previous solutions exist
    exist = 1
    a = pd.read_csv(pathcsv, index_col=0)
    a = a.sort_values('0', ascending=False) 
    del a['0']
    solutions = a[:50].to_numpy() # Import the set of solutions
    if solutions.size < 200:
        for s in range(50-(solutions.size/4)): # Initialize Initial Solutions
            solutions = np.append(solutions, np.array([[random.uniform(1, 1.9), random.uniform(0.1, 2), random.randint(7, 23), random.uniform(0.02, 0.15)]]), axis = 0)
else:
    for s in range(50): # Initialize Initial Solutions
        solutions = np.append(solutions, np.array([[random.uniform(1, 1.9), random.uniform(0.1, 2), random.randint(7, 23), random.uniform(0.02, 0.15)]]), axis = 0)

for i in range(2): # Generation Number  
    rankedsolutions = np.empty((0, 5))
    for s in solutions:
        print('===============')
        print(c)
        rankedsolutions = np.append(rankedsolutions, np.array([[fitness(s[0], s[1], int(s[2]), s[3]), s[0], s[1], int(s[2]), s[3]]]), axis = 0)
        c+=1 # Solution Number Tracker
    rankedsolutions = rankedsolutions[rankedsolutions[:, 0].argsort()] #Sort
    rankedsolutions = rankedsolutions[::-1] #Reverse
    
    print(f"=== Gen {i} best solutions ===")
    print(rankedsolutions[0])

    bestsolutions = rankedsolutions[:25]
    for k in bestsolutions: # Add the best solutions in 1 generation to generational run pool
        prev_bestsolutions = np.append(prev_bestsolutions, np.array([k]), axis=0)
        prev_bestsolutions = prev_bestsolutions[prev_bestsolutions[:, 0].argsort()] #Sort
        prev_bestsolutions = prev_bestsolutions[::-1] #Reverse
    
    elements = np.empty((0, 4))
    for s in prev_bestsolutions[:20]: # Select the inputs from best solutions
        elements = np.append(elements, np.array([[s[1], s[2], s[3], s[4]]]), axis=0)
        parents = np.unique(elements, axis=0)
        
    newGen = np.empty((0, 4))
    for _ in range(50): # Mutate the inputs
        e1 = random.choice(parents[:, 0])
        e2 = random.choice(parents[:, 1])
        e3 = random.choice(parents[:, 2])
        e4 = random.choice(parents[:, 3])
        newGen = np.append(newGen, np.array([[e1, e2, e3, e4]]), axis=0)
        
    for _ in range(int(np.rint(50 * mutation_rate))):
        rand = random.randint(0, 49)
        o1 = newGen[rand, 0] * random.uniform(0.98, 1.02)
        while o1 > 1.9 or o1 < 1:
            o1 = newGen[rand, 0] * random.uniform(0.98, 1.02)
        o2 = newGen[rand, 1] * random.uniform(0.98, 1.02)
        o3 = newGen[rand, 2] + random.randrange(-5, 5)
        while o3 > 23 or o3 < 7:
            o3 = newGen[rand, 2] + random.randrange(-5, 5)
        o4 = newGen[rand, 3] * random.uniform(0.98, 1.02)
        while o4 > 0.15 or o4 < 0.02:
            o4 = newGen[rand, 3] * random.uniform(0.98, 1.02)
        newGen[rand, 0] = o1
        newGen[rand, 1] = o2
        newGen[rand, 2] = o3
        newGen[rand, 3] = o4
        
    solutions = newGen

end = time.time() #Stop Timer
elapsed_time = end-start

print("=== Overall best solutions ===")
print(prev_bestsolutions[0])
print("=== Elapsed time ===")
print(elapsed_time)

elapsed_time = np.array([elapsed_time])

pd_pb = pd.DataFrame(data=prev_bestsolutions)
if exist == 1:
    pd_pb.to_csv(pathcsv, mode = 'a', header=False)
else:
    pd_pb.to_csv(pathcsv, mode = 'a', header=True)
    
convergence_data = prev_bestsolutions[:30]
avg_data = np.average(convergence_data, axis=0)
top1 = convergence_data[0]
error = np.absolute((top1-avg_data)/top1)
error_t = np.array([error]).T
error_t = error_t.T

pd_c = pd.DataFrame(data=error_t)
if os.path.isfile('inputs/convergence.csv') == 1:
    pd_c.to_csv('inputs/convergence.csv', mode = 'a', header=False)
else:
    pd_c.to_csv('inputs/convergence.csv', mode = 'a', header=True)
    
pd_e = pd.DataFrame(data=elapsed_time)
pd_e.to_csv('inputs/elapsed.csv', mode = 'a', header=False)

if np.average(error_t) < 0.001:
    visualization(int(prev_bestsolutions[0][3]))
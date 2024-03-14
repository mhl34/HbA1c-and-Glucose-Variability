import numpy as np
import pandas as pd

class GeneticAlgorithm:
    def __init__(self):
        self.num_iter = 100
        self.num_bits = 20
        self.num_pop = 100
        self.rate_cross = 0.9
        self.rate_mut = 1.0 / float(n_bits)
        
    # function: runs the genetic algorithm procedure
    # input: onemax, num_bits (number of bits), num_iter (number of iterations to go through), num_pop (number of animals in a population), rate_cross (rate of crossover), rate_mut (rate of mutation)
    # output: 
    def genetic_algorithm(self, num_bits, num_iter, num_pop, rate_cross, rate_mut):
        pop = self.initial_population(num_bits = self.num_bits, num_pop = self.num_pop)
        for gen in range(num_iter):
            scores = [objective_function(c) for c in pop]
    
    # function: get initial population of features to iterate through
    # input: num_bits (number of bits), num_pop (number of animals in a population)
    # output: the population in lists
    def initial_population(self, num_bits, num_pop):
        return [randint(0, 2, num_bits).tolist() for _ in range(num_pop)]
        
    # function: objective function
    # input: x
    # output: the return value of the objective function
    def objective_function(self, x):
        return -np.sum(x)
    
    # function: selection from the population
    # input: pop (population), scores (from the objective function), k (number of champions)
    # output: return the population of champions from the selection
    def selection(self, pop, scores, k = 3):
        
        
    # function: 
    # input: 
    # output: 
    def crossover(self, parent_1, parent_2, rate_cross):
        parent_1_copy = parent_1.copy()
        parent_2_copy = parent_2.copy()
        
    
    # function: run the algorithm
    # input: N/A
    # output: Analysis
    def run(self):
        pass
    
    
if __name__ == "__main__":
    obj = GeneticAlgorithm()
    obj.run()

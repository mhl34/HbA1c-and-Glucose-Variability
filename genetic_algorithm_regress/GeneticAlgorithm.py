import numpy as np
import pandas as pd
from models.Conv1DGeneticModel import Conv1DGeneticModel
import asyncio
import random
from procedures import train, evaluate, model_chooser

class GeneticAlgorithm:
    def __init__(self):
        self.num_iter = 100
        self.num_bits = 20
        self.num_pop = 100
        self.rate_cross = 0.9
        self.rate_mut = 1.0 / float(self.num_bits)
        
    # function: runs the genetic algorithm procedure
    # input: onemax, num_bits (number of bits), num_iter (number of iterations to go through), num_pop (number of animals in a population), rate_cross (rate of crossover), rate_mut (rate of mutation)
    # output: 
    def genetic_algorithm(self, num_bits, num_iter, num_pop, rate_cross, rate_mut):
        pop = self.initial_population(num_bits = num_bits, num_pop = num_pop)
        best, best_score = 0, self.objective_function(pop[0])
        for gen in range(num_iter):
            scores = [self.objective_function(c) for c in pop]
            for i in range(num_pop):
                if scores[i] > best_score:
                    best, best_score = pop[i], scores[i]
            selected = [self.selection(pop, scores) for _ in range(self.num_pop)]
            # create the next generation
            children = []
            for i in range(0, num_pop, 2):
                # get selected parents in pairs
                parent_1, parent_2 = selected[i], selected[i+1]
                # crossover and mutation
                for child in self.crossover(parent_1, parent_2, rate_cross):
                    # mutation
                    self.mutation(child, rate_mut)
                    # store for next generation
                    children.append(child)
            pop = children
        return best, best_score
    
    # function: get initial population of features to iterate through
    # input: num_bits (number of bits), num_pop (number of animals in a population)
    # output: the population in lists
    def initial_population(self, num_bits, num_pop):
        return [random.randint(0, 2, num_bits).tolist() for _ in range(num_pop)]
        
    # function: objective function (performing the machine learning model
    # input: x
    # output: the return value of the model accuracy
    def objective_function(self, x):
        pass
    
    # function: selection from the population
    # input: pop (population), scores (from the objective function), k
    # output: return the population of champions from the selection
    def selection(self, pop, scores, k = 3):
        selection_idx = random.randint(len(pop))
        for idx in random.randint(0, len(pop), k-1):
            if scores[idx] < scores[selection_idx]:
                selection_idx = idx
        return pop[selection_idx]
        
    # function: crossover between 2 different parents
    # input: parent_1 (bitstring), parent_2 (bitstring), rate_cross (rate of the crossover)
    # output: child_1 (crossed-over bitstring), child_2 (crossed-over bitstring)
    def crossover(self, parent_1, parent_2, rate_cross):
        parent_1_copy = parent_1.copy()
        parent_2_copy = parent_2.copy()
        # check for recombination
        if random.random() < rate_cross:
            # select crossover point that is not on the end of the string
            pt = random.randint(1, len(parent_1)-2)
            # perform crossover
            child_1 = parent_1_copy[:pt] + parent_2_copy[pt:]
            child_2 = parent_2_copy[:pt] + parent_1_copy[pt:]
        return child_1, child_2
        

    # function: performs a mutation
    # input: bitstring (bitstring of features selected), r_mut (rate of mutation)
    # output: NA
    def mutation(bitstring, r_mut):
        for idx in range(len(bitstring)):
            if random.random() < r_mut:
                bitstring[idx] = 1 - bitstring[idx]
    
    # function: run the algorithm
    # input: N/A
    # output: Analysis
    def run(self):
        pass
    
    # function: train the machine learning models with the various parameters
    async def train(self):
        pass
    
    async def evaluate(self):
        pass
    
    
if __name__ == "__main__":
    obj = GeneticAlgorithm()
    obj.run()

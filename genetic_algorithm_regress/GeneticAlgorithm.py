import numpy as np
import pandas as pd
from models.Conv1DGeneticModel import Conv1DGeneticModel
from utils.DataProcessor import DataProcessor
from utils.GeneticDataset import GeneticDataset
from utils.pp5 import pp5
import asyncio
import multiprocessing
import torch
from utils.procedures import train, evaluate, model_chooser

class GeneticAlgorithm:
    def __init__(self):
        # for CUDA multithreading
        torch.multiprocessing.set_start_method('spawn')

        # samples
        self.samples = [str(i).zfill(3) for i in range(1, 17)]
        self.trainSamples = self.samples[:-5]
        self.valSamples = self.samples[-5:]

        # genetic training regiment
        self.num_iter = 100
        self.num_bits = 24
        self.num_pop = 100
        self.rate_cross = 0.9
        self.rate_mut = 1.0 / float(self.num_bits)

        # model parameters
        self.dropout_p = 0.5
        self.normalize = False
        self.seq_len = 12
        self.dtype = torch.float64
        
        # os parameters
        self.main_dir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
        # self.main_dir = "/home/jovyan/work/physionet.org/files/big-ideas-glycemic-wearable/1.0.0/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {self.device}")

    def genetic_algorithm(self, num_bits, num_iter, num_pop, rate_cross, rate_mut):
        # Initialize the population
        pop = self.initial_population(num_bits=num_bits, num_pop=num_pop)
        # Initialize the best and best scores
        best, best_score = 0, self.objective_function(pop[0])
        
        pool = multiprocessing.Pool()
        
        # Iterate for the number of generations specified
        for _ in range(num_iter):
            # Get scores concurrently
            scores = pool.map(self.objective_function, pop)
            # iterate through the population
            for i in range(num_pop):
                # if the score is better than replace best by best score and the individual that scored the best
                if scores[i] > best_score:
                    best, best_score = pop[i], scores[i]
            # select based on the scores
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
        
        # Close the pool to release resources
        pool.close()
        
        return best, best_score
    
    # function: get initial population of features to iterate through
    # input: num_bits (number of bits), num_pop (number of animals in a population)
    # output: the population in lists
    def initial_population(self, num_bits, num_pop):
        return [np.random.randint(low = 0, high = 2, size = num_bits).tolist() for _ in range(num_pop)]
        
    # function: objective function (performing the machine learning model
    # input: x
    # output: the return value of the model accuracy
    def objective_function(self, x):
        num_features = sum(x)
        model = Conv1DGeneticModel(num_features = num_features, dropout_p = self.dropout_p, normalize = False, seq_len = self.seq_len, dtype = self.dtype).to(self.device)
        train(samples = self.trainSamples, model = model, featMetricList = x, main_dir = self.main_dir, dtype = self.dtype, device = self.device, seq_len = self.seq_len)
        loss_val = evaluate(samples = self.valSamples, model = model, featMetricList = x, main_dir = self.main_dir, dtype = self.dtype, device = self.device, seq_len = self.seq_len)
        return loss_val
    
    # function: selection from the population
    # input: pop (population), scores (from the objective function), k
    # output: return the population of champions from the selection
    def selection(self, pop, scores, k = 3):
        selection_idx = np.random.randint(len(pop))
        for idx in np.random.randint(low = 0, high = len(pop), size = k-1):
            if scores[idx] < scores[selection_idx]:
                selection_idx = idx
        return pop[selection_idx]
        
    # function: crossover between 2 different parents
    # input: parent_1 (bitstring), parent_2 (bitstring), rate_cross (rate of the crossover)
    # output: child_1 (crossed-over bitstring), child_2 (crossed-over bitstring)
    def crossover(self, parent_1, parent_2, rate_cross):
        parent_1_copy = parent_1.copy()
        parent_2_copy = parent_2.copy()
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()
        # check for recombination
        if np.random.random() < rate_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(low = 1, high = len(parent_1)-2)
            # perform crossover
            child_1 = parent_1_copy[:pt] + parent_2_copy[pt:]
            child_2 = parent_2_copy[:pt] + parent_1_copy[pt:]
        return child_1, child_2
        

    # function: performs a mutation
    # input: bitstring (bitstring of features selected), r_mut (rate of mutation)
    # output: NA
    def mutation(self, bitstring, r_mut):
        for idx in range(len(bitstring)):
            if np.random.random() < r_mut:
                bitstring[idx] = 1 - bitstring[idx]
    
    # function: run the algorithm
    # input: N/A
    # output: Analysis
    def run(self):
        print(self.genetic_algorithm(self.num_bits, self.num_iter, self.num_pop, self.rate_cross, self.rate_mut))
    
    
if __name__ == "__main__":
    obj = GeneticAlgorithm()
    obj.run()

from collections import defaultdict
from optimization import load_csv_and_filter, evaluate_fitness
from libraries import log
import random
import csv
import json
import multiprocessing

def initialize_population(pop_size, solution_size):
    # Initialize a population of random binary solutions using list comprehension
    return [[random.choice([0, 1]) for _ in range(solution_size)] for _ in range(pop_size)]

def select_parents(population, fitness_scores):
    # Implement Elitism: Select parents based on fitness (roulette wheel selection)
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    num_of_elitist_solution = round(len(population) * 0.2) # Determine the number of elite solutions (20%)
    index_of_max = fitness_scores.index(max(fitness_scores)) # Find the index of the best solution
    parents = random.choices(population, weights=probabilities, k=len(population) - num_of_elitist_solution) # Select parents using roulette wheel selection
    parents.extend([population[index_of_max]] * num_of_elitist_solution) # Keep the best solution from the previous generation
    return parents

def uniform_crossover(parent1, parent2, crossover_rate):
    return [random.choice([bit1, bit2]) if random.random() <= crossover_rate else bit1 for bit1, bit2 in zip(parent1, parent2)] # Perform uniform crossover

def mutate(solution, mutation_rate):
    return [bit if random.random() >= mutation_rate else 1 - bit for bit in solution] # Apply bit-flip mutation with a given mutation rate

def genetic_algorithm(pop_size, solution_size, mutation_rate, crossover_rate, train_file_paths,classifier_index, num_of_iterations, classes_file_path, num_of_packets_to_process, weights, log_file_path, max_num_of_generations, fields_file_path):
    pre_solutions = defaultdict(float)
    
    # Load classes
    with open(classes_file_path, 'r') as file:
        classes = json.loads(file.readline())

    # Load the packets
    log("loading packets...", log_file_path)
    packets_1 = []
    packets_2 = []

    # Read header from fields file
    with open(fields_file_path, 'r') as file:
        header = file.readline().strip().split(',') + ['label']
        packets_1.append(header)
        packets_2.append(header)

    packets_1.extend(element for element in load_csv_and_filter(classes, train_file_paths[0], num_of_packets_to_process, log_file_path))
    packets_2.extend(element for element in load_csv_and_filter(classes, train_file_paths[1], num_of_packets_to_process, log_file_path))

    log("", log_file_path)

    population = initialize_population(pop_size, solution_size)
    best_solution = None

    consecutive_same_solution_count = 0
    generation = 0

    while consecutive_same_solution_count < num_of_iterations and generation < max_num_of_generations:
        if best_solution is not None:
            parents = select_parents(population, fitness_scores) # Select parents for reproduction using Elitism
            new_population = [] # Create a new population through crossover and mutation

            while len(new_population) < pop_size:
                parent1, parent2 = random.choices(parents, k=2)
                child = uniform_crossover(parent1, parent2, crossover_rate)
                child = mutate(child, mutation_rate)
                new_population.append(child)

            population = new_population # Replace the old population with the new population
        
        # Evaluate the fitness of each solution in the population using multi-threading
        num_cores = multiprocessing.cpu_count() - 1 # Determine the number of CPU cores minus 1
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions, weights) for solution in population])

        pool.close()
        pool.join()

        fitness_scores = []
        for result in results:
            fitness_scores.append(result[0])
            pre_solutions.update(result[1])

        # Track and display the best solution in this generation
        new_best_solution = population[fitness_scores.index(max(fitness_scores))]
        if best_solution == new_best_solution:
            consecutive_same_solution_count += 1
        else:
            best_solution = new_best_solution
            consecutive_same_solution_count = 0

        sol_str = ''.join(map(str, best_solution))
        log(f"Generation {generation + 1}:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {max(fitness_scores)}", log_file_path)
        
        generation += 1

    log("", log_file_path)
    key = ''.join(map(str, best_solution))
    best_fitness = pre_solutions[key]

    return (best_solution, best_fitness)

def run(train_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations, fields_file_path):
    # Configuration parameters
    population_size = 50
    mutation_rate = 0.015
    crossover_rate = 0.5

    # Determine solution size (number of features)
    with open(train_file_paths[0], 'r') as file:
        solution_size = len(file.readline().split(',')) - 1

    return genetic_algorithm(
        population_size, solution_size, mutation_rate, crossover_rate, train_file_paths,
        classifier_index, num_of_iterations, classes_file_path, num_of_packets_to_process,
        weights, log_file_path, max_num_of_generations, fields_file_path
    )

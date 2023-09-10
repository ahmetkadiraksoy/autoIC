from collections import defaultdict
from optimization import load_csv, evaluate_fitness
import random
import csv
import concurrent.futures
import os
import threading
import random
import json

# Define a lock for synchronization
thread_lock = threading.Lock()

def initialize_population(pop_size, solution_size):
    # Initialize a population of random binary solutions
    population = []
    for _ in range(pop_size):
        solution = [random.choice([0, 1]) for _ in range(solution_size)]
        population.append(solution)
    return population

def select_parents(population, fitness_scores):
    # Implement Elitism: Select parents based on fitness (roulette wheel selection)
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    num_of_elitist_solution = round(len(population) * 0.2) # make 20% the same as the best solution from previous population
    parents = random.choices(population, weights=probabilities, k=len(population) - num_of_elitist_solution)
    # Keep the best solution from the previous generation
    index_of_max = max((num, index) for index, num in enumerate(fitness_scores))[1]
    for _ in range(num_of_elitist_solution):
        parents.append(population[index_of_max])
    return parents

def uniform_crossover(parent1, parent2, crossover_rate):
    # Perform uniform crossover
    if random.random() <= crossover_rate:
        child = [random.choice([bit1, bit2]) for bit1, bit2 in zip(parent1, parent2)]
        return child
    else:
        return parent1  # No crossover

def mutate(solution, mutation_rate):
    # Apply bit-flip mutation with a given mutation rate
    mutated_solution = [bit ^ (random.random() < mutation_rate) for bit in solution]
    return mutated_solution

def randomize_packets(list_of_lists):
    # Separate the first line (header) from the rest of the data
    header = list_of_lists[0]
    data = list_of_lists[1:]

    # Shuffle the data portion (excluding the header)
    random.shuffle(data)

    # Combine the header and shuffled data to create the final randomized list
    return [header] + data

def genetic_algorithm(pop_size, solution_size, mutation_rate, crossover_rate, fitness_function_file_paths, classifier_index, pre_solutions, num_of_iterations, classes_file_path, num_of_packets_to_process, weights):
    # Load classes
    with open(classes_file_path, 'r') as file:
        classes = json.loads(file.readline())

    # Load the packets
    print("loading packets...")
    packets_1 = []
    packets_2 = []

    # Read header from CSV
    with open(fitness_function_file_paths[0], 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        packets_1.append(header)
        packets_2.append(header)

    packets_1.extend(element for element in load_csv(classes, fitness_function_file_paths[0], num_of_packets_to_process))
    packets_2.extend(element for element in load_csv(classes, fitness_function_file_paths[1], num_of_packets_to_process))

    print()

    population = initialize_population(pop_size, solution_size)
    best_solution = None

    consecutive_same_solution_count = 0
    generation = 1

    while consecutive_same_solution_count < num_of_iterations:
        if best_solution is not None:
            # Select parents for reproduction using Elitism
            parents = select_parents(population, fitness_scores)

            # Create a new population through crossover and mutation
            new_population = []
            while len(new_population) < pop_size:
                parent1, parent2 = random.choices(parents, k=2)
                child = uniform_crossover(parent1, parent2, crossover_rate)
                child = mutate(child, mutation_rate)
                new_population.append(child)
        
            # Replace the old population with the new population
            population = new_population
        
        # Evaluate the fitness of each solution in the population using multi-threading
        fitness_scores = []
        num_cores = os.cpu_count() - 1 # Determine the number of CPU cores minus 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            for solution in population:
                future = executor.submit(evaluate_fitness, solution, packets_1, packets_2, classifier_index, pre_solutions, weights)
                fitness_scores.append(future.result())

        # Track and display the best solution in this generation
        new_best_solution = population[fitness_scores.index(max(fitness_scores))]
        if best_solution == new_best_solution:
            consecutive_same_solution_count += 1
        else:
            best_solution = new_best_solution
            consecutive_same_solution_count = 0
        print(f"Generation {generation}:\t[{''.join(map(str, best_solution))}]\tFitness: {max(fitness_scores)}")
        
        generation += 1
    print()
    key = ''.join(map(str, best_solution))
    best_fitness = pre_solutions[key]

    return best_solution, best_fitness

def run(fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights):
    population_size = 50
    mutation_rate = 0.015
    crossover_rate = 0.5
    pre_solutions = defaultdict(float)

    # Determine solution size (number of features)
    with open(fitness_function_file_paths[0], 'r') as file:
        first_line = file.readline()
    solution_size = len(first_line.split(',')) - 1

    return genetic_algorithm(population_size, solution_size, mutation_rate, crossover_rate, fitness_function_file_paths, classifier_index, pre_solutions, num_of_iterations, classes_file_path, num_of_packets_to_process, weights)

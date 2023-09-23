from collections import defaultdict
from optimization import load_csv, evaluate_fitness
from libraries import log
import random
import csv
import random
import json
import multiprocessing

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

def genetic_algorithm(pop_size, solution_size, mutation_rate, crossover_rate, train_file_paths, classifier_index, num_of_iterations, classes_file_path, num_of_packets_to_process, weights, log_file_path, max_num_of_generations):
    pre_solutions = defaultdict(float)
    
    # Load classes
    with open(classes_file_path, 'r') as file:
        classes = json.loads(file.readline())

    # Load the packets
    log("loading packets...", log_file_path)
    packets_1 = []
    packets_2 = []

    # Read header from CSV
    with open(train_file_paths[0], 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        packets_1.append(header)
        packets_2.append(header)

    packets_1.extend(element for element in load_csv(classes, train_file_paths[0], num_of_packets_to_process, log_file_path))
    packets_2.extend(element for element in load_csv(classes, train_file_paths[1], num_of_packets_to_process, log_file_path))

    log("", log_file_path)

    population = initialize_population(pop_size, solution_size)
    best_solution = None

    consecutive_same_solution_count = 0
    generation = 0

    while consecutive_same_solution_count < num_of_iterations and generation < max_num_of_generations:
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

def run(train_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations):
    population_size = 50
    mutation_rate = 0.015
    crossover_rate = 0.5

    # Determine solution size (number of features)
    with open(train_file_paths[0], 'r') as file:
        first_line = file.readline()
    solution_size = len(first_line.split(',')) - 1

    return genetic_algorithm(population_size, solution_size, mutation_rate, crossover_rate, train_file_paths, classifier_index, num_of_iterations, classes_file_path, num_of_packets_to_process, weights, log_file_path, max_num_of_generations)

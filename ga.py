import random
import csv
import ml
import concurrent.futures
import os
import threading
import sys
from collections import defaultdict

# Define a lock for synchronization
ga_solutions_lock = threading.Lock()

# Load and prepare data
def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def initialize_population(pop_size, solution_size):
    # Initialize a population of random binary solutions
    population = []
    for _ in range(pop_size):
        solution = [random.choice([0, 1]) for _ in range(solution_size)]
        population.append(solution)
    return population

def evaluate_fitness(solution, packets_1, packets_2, clf, ga_solutions):
    # If no features are to be selected
    if sum(solution) == 0:
        return 0.0

    key = ''.join(map(str, solution))

    # Acquire the lock before reading ga_solutions
    with ga_solutions_lock:
        if key in ga_solutions:
            return ga_solutions[key]

    # Append 1 to the end so that it doesn't filter out the 'class' column
    solution_new = list(solution)
    solution_new.append(1)

    # Filter features
    filtered_packets_1 = [[col for col, m in zip(row, solution_new) if m] for row in packets_1]
    filtered_packets_2 = [[col for col, m in zip(row, solution_new) if m] for row in packets_2]
    
    fitness_1 = ml.classify(filtered_packets_1, filtered_packets_2, clf)
    fitness_2 = ml.classify(filtered_packets_2, filtered_packets_1, clf)

    average_accuracy = (fitness_1 + fitness_2) / 2.0

    # Calculate feature accuracy
    num_selected_features = sum(solution)
    total_features = len(solution) - 1  # Excluding the class column

    feature_accuracy = 1 - ((num_selected_features - 1) / total_features)

    # Calculate fitness as a weighted combination of average accuracy and feature accuracy
    fitness = 0.9 * average_accuracy + 0.1 * feature_accuracy

    # Acquire the lock before updating ga_solutions
    with ga_solutions_lock:
        ga_solutions[key] = fitness

    return fitness

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

def genetic_algorithm(pop_size, solution_size, num_generations, mutation_rate, crossover_rate, packets_1_location, packets_2_location, clf, ga_solutions, num_of_iterations):
    # Load the packets
    packets_1 = load_csv(packets_1_location)
    packets_2 = load_csv(packets_2_location)

    population = initialize_population(pop_size, solution_size)
    best_solution = None

    consecutive_same_solution_count = 0
    generation = 1

    while generation <= num_generations:
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
                future = executor.submit(evaluate_fitness, solution, packets_1, packets_2, clf, ga_solutions)
                fitness_scores.append(future.result())

        # Track and display the best solution in this generation
        new_best_solution = population[fitness_scores.index(max(fitness_scores))]
        if best_solution == new_best_solution:
            consecutive_same_solution_count += 1
        else:
            best_solution = new_best_solution
            consecutive_same_solution_count = 0
        print(f"Generation {generation}:\t[{''.join(map(str, best_solution))}]\tFitness: {max(fitness_scores)}")
        
        # Terminate if the same solution has been observed consecutively for a specified number of times
        if consecutive_same_solution_count >= num_of_iterations:
            break

        generation += 1

    key = ''.join(map(str, best_solution))
    best_fitness = ga_solutions[key]

    return best_solution, best_fitness

def run(packets_1_location, packets_2_location, clf):
    # Example usage:
    population_size = 50
    num_generations = 100
    mutation_rate = 0.015
    crossover_rate = 0.5
    num_of_iterations = 10
    ga_solutions = defaultdict(float)

    # Determine solution size (number of features)
    with open(packets_1_location, 'r') as file:
        first_line = file.readline()
    solution_size = len(first_line.split(',')) - 1

    return genetic_algorithm(population_size, solution_size, num_generations, mutation_rate, crossover_rate, packets_1_location, packets_2_location, clf, ga_solutions, num_of_iterations)

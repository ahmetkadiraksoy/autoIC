import random
import threading
import csv
import ml
from collections import defaultdict
import sys

# Define a lock for synchronization
ga_solutions_lock = threading.Lock()

# Load and prepare data
def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

# Define the fitness function
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

# Define the ACO algorithm
def ant_colony_optimization(n_ants, n_iterations, pheromone_decay, pheromone_strength, lock, packets_1_location, packets_2_location, clf, solution_size, ga_solutions):
    # Load the packets
    packets_1 = load_csv(packets_1_location)
    packets_2 = load_csv(packets_2_location)

    # Initialize the pheromone matrix with equal values for each ant
    pheromones = [1.0] * n_ants

    # Define the ant behavior
    def ant_behavior(ant_index, solutions, fitness_values):
        # Each ant chooses a solution based on the pheromone levels
        # and the fitness function, and deposits pheromones accordingly
        solution = solutions[ant_index]
        fitness_value = fitness_values[ant_index]
        lock.acquire()
        pheromones[ant_index] += pheromone_strength * fitness_value  # Increase pheromone based on fitness
        pheromones[ant_index] *= pheromone_decay  # Decay pheromone
        lock.release()

    # Run the algorithm until the same best solution is produced 'n_iterations' times in a row
    best_solution_counter = 0
    iteration_counter = 0
    best_solution = None
    best_fitness = None
    best_ones = None

    while best_solution_counter + 1 < n_iterations:
        threads = []
        solutions = [[random.randint(0, 1) for _ in range(solution_size)] for _ in range(n_ants)]
        fitness_values = [evaluate_fitness(s, packets_1, packets_2, clf, ga_solutions) for s in solutions]

        current_best_solution = None
        current_best_fitness = None
        current_best_ones = None

        for j in range(n_ants):
            t = threading.Thread(target=ant_behavior, args=(j, solutions, fitness_values))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Choose the best solution based on the fitness values
        ant_fitness_max_index = fitness_values.index(max(fitness_values))
        current_best_solution = solutions[ant_fitness_max_index]
        current_best_fitness = evaluate_fitness(current_best_solution, packets_1, packets_2, clf, ga_solutions)
        current_best_ones = sum(current_best_solution)

        # If the current best solution is better than the previous best solution,
        # update the best solution and best fitness
        if best_solution is None or current_best_fitness > evaluate_fitness(best_solution, packets_1, packets_2, clf, ga_solutions):
            best_solution = current_best_solution
            best_fitness = current_best_fitness
            best_ones = current_best_ones
            best_solution_counter = 0
        else:
            best_solution_counter += 1

        iteration_counter += 1

        # Print current best solution with a grid of filled squares for 1 and empty squares for 0
        print(f"Generation {iteration_counter}:\t[{''.join(map(str, best_solution))}]\tFitness: {best_fitness}")

    # Return the best solution and its fitness value
    return (best_solution, best_fitness)

def run(packets_1_location, packets_2_location, clf):
    n_ants = 10
    n_iterations = 10
    pheromone_strength = 1
    pheromone_decay = 0.5
    lock = threading.Lock()
    ga_solutions = defaultdict(float)

    # Determine solution size (number of features)
    with open(packets_1_location, 'r') as file:
        first_line = file.readline()
    solution_size = len(first_line.split(',')) - 1

    return ant_colony_optimization(n_ants, n_iterations, pheromone_decay, pheromone_strength, lock, packets_1_location, packets_2_location, clf, solution_size, ga_solutions)

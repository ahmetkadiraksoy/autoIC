from collections import defaultdict
from optimization import load_csv, evaluate_fitness
from libraries import log
import random
import threading
import csv
import json
import multiprocessing

# Define a lock for synchronization
thread_lock = threading.Lock()

# Define the ACO algorithm
def ant_colony_optimization(num_of_ants, num_of_iterations, pheromone_decay, pheromone_strength, fitness_function_file_paths, classifier_index, solution_size, pre_solutions, classes_file_path, num_of_packets_to_process, weights, log_file_path, max_num_of_generations):
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

    log("", log_file_path)

    # Initialize the pheromone matrix with equal values for each ant
    pheromones = [1.0] * num_of_ants

    # Define the ant behavior
    def ant_behavior(ant_index, fitness_values):
        fitness_value = fitness_values[ant_index]

        with thread_lock:
            pheromones[ant_index] += pheromone_strength * fitness_value  # Increase pheromone based on fitness
            pheromones[ant_index] *= pheromone_decay  # Decay pheromone

    # Run the algorithm until the same best solution is produced 'n_iterations' times in a row
    best_solution_counter = 1
    iteration_counter = 0
    best_solution = None
    best_fitness = None

    while best_solution_counter < num_of_iterations and best_solution_counter < max_num_of_generations:
        threads = []
        solutions = [[random.randint(0, 1) for _ in range(solution_size)] for _ in range(num_of_ants)]

        num_cores = multiprocessing.cpu_count() - 1 # Determine the number of CPU cores minus 1
        with multiprocessing.Pool(processes=num_cores) as pool:
            fitness_values = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions, weights) for solution in solutions])

        pool.close()
        pool.join()

        current_best_solution = None
        current_best_fitness = None

        for j in range(num_of_ants):
            t = threading.Thread(target=ant_behavior, args=(j, fitness_values))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Choose the best solution based on the fitness values
        ant_fitness_max_index = fitness_values.index(max(fitness_values))
        current_best_solution = solutions[ant_fitness_max_index]
        current_best_fitness = evaluate_fitness(current_best_solution, packets_1, packets_2, classifier_index, pre_solutions, weights)

        # If the current best solution is better than the previous best solution,
        # update the best solution and best fitness
        if best_solution is None or current_best_fitness > evaluate_fitness(best_solution, packets_1, packets_2, classifier_index, pre_solutions, weights):
            best_solution = current_best_solution
            best_fitness = current_best_fitness
            best_solution_counter = 1
        else:
            best_solution_counter += 1

        iteration_counter += 1

        # Print current best solution with a grid of filled squares for 1 and empty squares for 0
        log(f"Generation {iteration_counter}:\t[{''.join(map(str, best_solution))}]\tFitness: {best_fitness}", log_file_path)
    log("", log_file_path)

    # Return the best solution and its fitness value
    return (best_solution, best_fitness)

def run(fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations):
    num_of_ants = 10
    pheromone_strength = 1
    pheromone_decay = 0.5
    pre_solutions = defaultdict(float)

    # Determine solution size (number of features)
    with open(fitness_function_file_paths[0], 'r') as file:
        first_line = file.readline()
    solution_size = len(first_line.split(',')) - 1

    return ant_colony_optimization(num_of_ants, num_of_iterations, pheromone_decay, pheromone_strength, fitness_function_file_paths, classifier_index, solution_size, pre_solutions, classes_file_path, num_of_packets_to_process, weights, log_file_path, max_num_of_generations)

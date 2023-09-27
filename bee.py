from collections import defaultdict
from optimization import load_csv_and_filter, evaluate_fitness
from libraries import log
import random
import threading
import json

# Define the ABC algorithm
def artificial_bee_colony(num_of_bees, num_of_iterations, limit, limit_inc, limit_dec, solution_size, fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, weights, pre_solutions, log_file_path):
    # Load classes
    with open(classes_file_path, 'r') as file:
        classes = json.loads(file.readline())

    # Load the packets
    log("Loading packets...", log_file_path)
    packets_1 = load_csv_and_filter(classes, fitness_function_file_paths[0], num_of_packets_to_process)
    packets_2 = load_csv_and_filter(classes, fitness_function_file_paths[1], num_of_packets_to_process)

    # Initialize the bee population
    bees = [{'solution': [random.randint(0, 1) for _ in range(solution_size)], 'fitness': None} for _ in range(num_of_bees)]

    # Initialize the limit parameter
    current_limit = limit

    # Run the algorithm for a fixed number of iterations
    for iteration in range(num_of_iterations):
        for bee in bees:
            # Evaluate the fitness of the bee's solution if it hasn't been evaluated before
            if bee['fitness'] is None:
                bee['fitness'] = evaluate_fitness(bee['solution'], packets_1, packets_2, classifier_index, pre_solutions, weights)

            # Generate a new solution by perturbing the current solution
            new_solution = [bit ^ (random.random() < 0.1) for bit in bee['solution']]

            # Evaluate the fitness of the new solution
            new_fitness = evaluate_fitness(new_solution, packets_1, packets_2, classifier_index, pre_solutions, weights)

            # Update the bee's solution if the new solution is better
            if new_fitness > bee['fitness']:
                bee['solution'] = new_solution
                bee['fitness'] = new_fitness
                current_limit = limit  # Reset the limit
            else:
                current_limit += limit_inc

            # Perform scout bee phase if the limit is exceeded
            if current_limit >= limit_dec:
                bee['solution'] = [random.randint(0, 1) for _ in range(solution_size)]
                bee['fitness'] = None
                current_limit = limit

        # Find and print the best solution in the current iteration
        best_bee = max(bees, key=lambda bee: bee['fitness'])
        log(f"Iteration {iteration + 1}:\t[{''.join(map(str, best_bee['solution']))}]\tFitness: {best_bee['fitness']}", log_file_path)

    # Return the best solution found
    best_solution = max(bees, key=lambda bee: bee['fitness'])
    return best_solution['solution'], best_solution['fitness']

def run(fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path):
    num_of_bees = 10
    limit = 10
    limit_inc = 1
    limit_dec = 5
    pre_solutions = defaultdict(float)

    # Determine solution size (number of features)
    with open(fitness_function_file_paths[0], 'r') as file:
        solution_size = len(file.readline().split(',')) - 1

    return artificial_bee_colony(
        num_of_bees, num_of_iterations, limit, limit_inc, 
        limit_dec, solution_size, fitness_function_file_paths, 
        classifier_index, classes_file_path, num_of_packets_to_process, 
        weights, pre_solutions, log_file_path
    )

from collections import defaultdict
import multiprocessing
from optimization import load_csv_and_filter, evaluate_fitness
from libraries import log
import random
import sys
import json

def initialize_solution(solution_size):
    # Helper function to initialize a random solution
    return [random.randint(0, 1) for _ in range(solution_size)]

def generate_neighbor_solution(solution):
    # Helper function to generate a neighbor solution by flipping a random bit
    neighbor_solution = solution.copy()
    index_to_flip = random.randint(0, len(neighbor_solution) - 1)
    neighbor_solution[index_to_flip] = 1 - neighbor_solution[index_to_flip]
    return neighbor_solution

def artificial_bee_colony(solution_size, num_of_iterations, train_file_paths, classifier_index, num_of_packets_to_process, weights, log_file_path, fields_file_path, num_of_employed_bees, num_of_onlooker_bees, num_of_scout_bees, classes_file_path, num_cores):
    pre_solutions = defaultdict(float)
    
    # Load classes
    try:
        with open(classes_file_path, 'r') as file:
            classes = json.loads(file.readline())
    except FileNotFoundError:
        print(f"The file {classes_file_path} does not exist.")
        sys.exit(1)

    # Load the packets
    log("loading packets...", log_file_path)
    packets_1 = []
    packets_2 = []

    # Read header from fields file
    try:
        with open(fields_file_path, 'r') as file:
            header = file.readline().strip().split(',') + ['label']
            packets_1.append(header)
            packets_2.append(header)
    except FileNotFoundError:
        print(f"The file {fields_file_path} does not exist.")
        sys.exit(1)

    packets_1.extend(element for element in load_csv_and_filter(classes, train_file_paths[0], num_of_packets_to_process, log_file_path))
    packets_2.extend(element for element in load_csv_and_filter(classes, train_file_paths[1], num_of_packets_to_process, log_file_path))

    log("", log_file_path)

    # Initialize employed bees with random solutions
    employed_bees = [initialize_solution(solution_size) for _ in range(num_of_employed_bees)]
    best_solution = None
    best_fitness = None

    consecutive_same_solution_count = 0
    generation = 0

    while consecutive_same_solution_count < num_of_iterations:
        neighbor_bees = [generate_neighbor_solution(employed_bees[i]) for i in range(num_of_employed_bees)]

        # Precompute fitness values for all employed bees
        with multiprocessing.Pool(processes=num_cores) as pool:
            results_precompute = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions, weights) for solution in employed_bees])

        pool.close()
        pool.join()

        employed_bees_fitness = []
        for result in results_precompute:
            employed_bees_fitness.append(result[0])

        with multiprocessing.Pool(processes=num_cores) as pool:
            results_employed = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions, weights) for solution in neighbor_bees])

        pool.close()
        pool.join()

        neighbor_solution_fitness = []
        for result in results_employed:
            neighbor_solution_fitness.append(result[0])

        # Employed bees phase: Search for new solutions
        for i in range(num_of_employed_bees):
            if neighbor_solution_fitness[i] > employed_bees_fitness[i]:
                employed_bees[i] = neighbor_bees[i]
                employed_bees_fitness[i] = neighbor_solution_fitness[i]

        # Onlooker bees phase: Select solutions to follow
        onlooker_bees = []
        total_fitness = sum(employed_bees_fitness)

        for i in range(num_of_onlooker_bees):
            roulette_wheel = random.uniform(0, total_fitness)
            cum_fitness = 0
            for j in range(num_of_employed_bees):
                cum_fitness += employed_bees_fitness[j]
                if cum_fitness >= roulette_wheel:
                    onlooker_bees.append(employed_bees[j])
                    break

        # Scout bees phase: Replace solutions with low fitness
        for i in range(num_of_scout_bees):
            solution = initialize_solution(solution_size)
            employed_bees[random.randint(0, num_of_employed_bees - 1)] = solution

        # Evaluate fitness for the best solution and display results
        best_solution_index = employed_bees_fitness.index(max(employed_bees_fitness))
        best_solution = employed_bees[best_solution_index]

        best_fitness = employed_bees_fitness[best_solution_index]
        sol_str = ''.join(map(str, best_solution))

        log(f"Generation {generation + 1}:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}", log_file_path)

        generation += 1

    log("", log_file_path)

    return (best_solution, best_fitness)

def run(train_file_paths, fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, fields_file_path, num_cores):
    num_of_employed_bees = 10
    num_of_onlooker_bees = 10
    num_of_scout_bees = 10

    # Determine solution size (number of features)
    with open(fitness_function_file_paths[0], 'r') as file:
        solution_size = len(file.readline().split(',')) - 1

    return artificial_bee_colony(
        solution_size, num_of_iterations, train_file_paths, classifier_index,
        num_of_packets_to_process, weights, log_file_path, fields_file_path,
        num_of_employed_bees, num_of_onlooker_bees, num_of_scout_bees,
        classes_file_path, num_cores)


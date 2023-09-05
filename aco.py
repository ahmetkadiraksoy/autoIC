import random
import csv
import ml
import numpy as np
from collections import defaultdict
import concurrent.futures
import threading
import os

# Define a lock for synchronization
ga_solutions_lock = threading.Lock()

# Load and prepare data
def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def evaluate_fitness(solution, packets_1, packets_2, clf, ga_solutions):
    if not any(solution):
        return 0.0

    key = ''.join(map(str, solution))

    # Acquire the lock before reading ga_solutions
    with ga_solutions_lock:
        if key in ga_solutions:
            return ga_solutions[key]

    solution_new = solution + [1]

    filtered_packets_1 = [[col for col, m in zip(row, solution_new) if m] for row in packets_1]
    filtered_packets_2 = [[col for col, m in zip(row, solution_new) if m] for row in packets_2]

    fitness_1 = ml.classify(filtered_packets_1, filtered_packets_2, clf)
    fitness_2 = ml.classify(filtered_packets_2, filtered_packets_1, clf)

    average_accuracy = (fitness_1 + fitness_2) / 2.0

    num_selected_features = sum(solution)
    total_features = len(solution) - 1
    feature_accuracy = 1 - ((num_selected_features - 1) / total_features)

    fitness = 0.9 * average_accuracy + 0.1 * feature_accuracy

    # Acquire the lock before updating ga_solutions
    with ga_solutions_lock:
        ga_solutions[key] = fitness

    return fitness

def generate_ant_solution(solution_size, current_node, ant_solution, pheromone_matrix, packets_1, packets_2, clf, alpha, beta, ga_solutions):
    probabilities = []
    total_probability = 0.0

    for feature in range(solution_size):
        if feature == current_node or ant_solution[feature] == 1:
            probabilities.append(0.0)
        else:
            pheromone = pheromone_matrix[current_node][feature]
            heuristic = evaluate_fitness(ant_solution[:feature] + [1] + ant_solution[feature + 1:], packets_1, packets_2, clf, ga_solutions)
            probability = (pheromone ** alpha) * (heuristic ** beta)
            probabilities.append(probability)
            total_probability += probability

    if total_probability == 0.0:
        return random.choice([i for i, prob in enumerate(probabilities) if prob == 0.0])

    normalized_probabilities = [prob / total_probability for prob in probabilities]
    selected_feature = np.random.choice(range(solution_size), p=normalized_probabilities)

    return selected_feature

def update_pheromone(pheromone_matrix, ant_population, fitness_scores, rho, q, solution_size, population_size):
    pheromone_delta = np.zeros((solution_size, solution_size))

    for i in range(population_size):
        ant_solution = ant_population[i]
        fitness = fitness_scores[i]

        for j in range(solution_size):
            if ant_solution[j] == 1:
                pheromone_delta[j][j] += q / fitness

    pheromone_matrix = (1 - rho) * pheromone_matrix + pheromone_delta
    return pheromone_matrix

def ant_colony_optimization(population_size, solution_size, num_generations, alpha, beta, rho, q, packets_1_location, packets_2_location, clf, num_of_iterations, ga_solutions):
    # Load the packets
    packets_1 = load_csv(packets_1_location)
    packets_2 = load_csv(packets_2_location)

    pheromone_matrix = np.ones((solution_size, solution_size))
    best_solution = None
    best_fitness = 0.0

    consecutive_same_solution_count = 0

    for generation in range(num_generations):
        ant_population = []

        for _ in range(population_size):
            current_node = 0
            ant_solution = [0] * solution_size

            for _ in range(solution_size):
                next_node = generate_ant_solution(solution_size, current_node, ant_solution, pheromone_matrix, packets_1, packets_2, clf, alpha, beta, ga_solutions)
                ant_solution[next_node] = 1
                current_node = next_node

            ant_population.append(ant_solution)

        old_best_solution = best_solution

        # Evaluate the fitness of each solution in the population using multi-threading
        fitness_scores = []
        num_cores = os.cpu_count() - 1 # Determine the number of CPU cores minus 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            for solution in ant_population:
                future = executor.submit(evaluate_fitness, solution, packets_1, packets_2, clf, ga_solutions)
                fitness_scores.append(future.result())

        max_fitness_index = np.argmax(fitness_scores)

        if fitness_scores[max_fitness_index] > best_fitness:
            best_solution = ant_population[max_fitness_index]
            best_fitness = fitness_scores[max_fitness_index]

        if old_best_solution == None:
            old_best_solution = best_solution

        pheromone_matrix = update_pheromone(pheromone_matrix, ant_population, fitness_scores, rho, q, solution_size, population_size)

        if best_solution == old_best_solution:
            consecutive_same_solution_count += 1
        else:
            consecutive_same_solution_count = 0
        print(f"Generation {generation + 1}: {best_solution}, Fitness: {best_fitness}")

        # Terminate if the same solution has been observed consecutively for a specified number of times
        if consecutive_same_solution_count >= num_of_iterations:
            break

    return best_solution, best_fitness

def run(packets_1_location, packets_2_location, clf):
    # Example usage:
    population_size = 50
    num_generations = 100
    num_of_iterations = 10
    alpha = 1.0
    beta = 1.0
    rho = 0.1
    q = 1.0
    ga_solutions = defaultdict(float)

    # Determine solution size (number of features)
    with open(packets_1_location, 'r') as file:
        first_line = file.readline()
    solution_size = len(first_line.split(',')) - 1

    return ant_colony_optimization(population_size, solution_size, num_generations, alpha, beta, rho, q, packets_1_location, packets_2_location, clf, num_of_iterations, ga_solutions)

from collections import defaultdict
import threading
import ml
import csv
import random
import numpy as np

# Define a lock for synchronization
thread_lock = threading.Lock()

def evaluate_fitness(solution, packets_1, packets_2, classifier_index, pre_solutions, weights):
    pre_solutions_temp = defaultdict(float)

    # If no features are to be selected
    if sum(solution) == 0:
        return 0.0, pre_solutions_temp

    key = ''.join(map(str, solution))

    # Acquire the lock before reading pre_solutions
    if key in pre_solutions:
        return pre_solutions[key], pre_solutions_temp

    # Append 1 to the end so that it doesn't filter out the 'class' column
    solution_new = solution + [1]

    # Filter features
    filtered_packets_1 = [[col for col, m in zip(row, solution_new) if m] for row in packets_1]
    filtered_packets_2 = [[col for col, m in zip(row, solution_new) if m] for row in packets_2]
    
    fitness_1 = ml.classify(filtered_packets_1, filtered_packets_2, classifier_index)[0]
    fitness_2 = ml.classify(filtered_packets_2, filtered_packets_1, classifier_index)[0]

    average_accuracy = np.mean([fitness_1, fitness_2])

    # Calculate feature accuracy
    num_selected_features = sum(solution)
    total_features = len(solution) - 1  # Excluding the class column

    feature_accuracy = 1 - ((num_selected_features - 1) / total_features)

    # Calculate fitness as a weighted combination of average accuracy and feature accuracy
    fitness = weights[0] * average_accuracy + weights[1] * feature_accuracy

    # Acquire the lock before updating pre_solutions
    pre_solutions_temp[key] = fitness

    return fitness, pre_solutions_temp

def load_csv(classes, fitness_function_file_path, n):
    packets = []
    for i in range(len(classes)):
        print("reading from " + classes[str(i)] + "...")
        with open(fitness_function_file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)

            # Skip the header row (if it exists)
            next(csv_reader, None)

            lines = []
            # Iterate through the rows line by line
            for row in csv_reader:
                if row[-1] == str(i):
                    lines.append(row)

            random.shuffle(lines)

            if n == 0:
                no_of_packets_to_keep = len(lines)
            else:
                no_of_packets_to_keep = min(n, len(lines))

            # Append the selected lines directly to packets
            packets.extend(lines[:no_of_packets_to_keep])
    return packets

import threading
import ml
import csv
import random
import numpy as np

# Define a lock for synchronization
thread_lock = threading.Lock()

def evaluate_fitness(solution, packets_1, packets_2, classifier_index, pre_solutions, weights):
    # If no features are to be selected
    if sum(solution) == 0:
        return 0.0

    key = ''.join(map(str, solution))

    # Acquire the lock before reading pre_solutions
    with thread_lock:
        if key in pre_solutions:
            return pre_solutions[key]

    # Append 1 to the end so that it doesn't filter out the 'class' column
    solution_new = solution + [1]

    # Filter features
    filtered_packets_1 = [[col for col, m in zip(row, solution_new) if m] for row in packets_1]
    filtered_packets_2 = [[col for col, m in zip(row, solution_new) if m] for row in packets_2]
    
    fitness_1 = ml.classify(filtered_packets_1, filtered_packets_2, classifier_index)
    fitness_2 = ml.classify(filtered_packets_2, filtered_packets_1, classifier_index)

    average_accuracy = np.mean([fitness_1, fitness_2])

    # Calculate feature accuracy
    num_selected_features = sum(solution)
    total_features = len(solution) - 1  # Excluding the class column

    feature_accuracy = 1 - ((num_selected_features - 1) / total_features)

    # Calculate fitness as a weighted combination of average accuracy and feature accuracy
    fitness = weights[0] * average_accuracy + weights[1] * feature_accuracy

    # Acquire the lock before updating pre_solutions
    with thread_lock:
        pre_solutions[key] = fitness

    return fitness

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

# def load_csv(classes, fitness_function_file_path, n):
#     class_packets = {}  # Use a dictionary to store unique lines for each class

#     with open(fitness_function_file_path, 'r', newline='') as csv_file:
#         csv_reader = csv.reader(csv_file)

#         # Skip the header row (if it exists)
#         header = next(csv_reader, None)
        
#         if header is None:
#             return []  # No data in the file
        
#         # Find the index of the class label column
#         class_label_index = len(header) - 1

#         # Read all lines and shuffle them
#         lines = list(csv_reader)
#         random.shuffle(lines)

#         for i in range(len(classes)):
#             print("reading from " + classes[str(i)] + "...")
#             class_lines = []

#             # Iterate through the shuffled lines
#             for row in lines:
#                 if row[class_label_index] == str(i):
#                     class_lines.append(row)

#             if n == 0:
#                 no_of_packets_to_keep = len(class_lines)
#             else:
#                 no_of_packets_to_keep = min(n, len(class_lines))

#             # Store the selected lines for this class
#             class_packets[i] = class_lines[:no_of_packets_to_keep]

#     # Flatten the dictionary values into a list of packets
#     packets = [packet for class_lines in class_packets.values() for packet in class_lines]

#     return packets

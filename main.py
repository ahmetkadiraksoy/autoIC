from sklearn.metrics import classification_report
from libraries import log
import pandas as pd
import subprocess
import csv
import sys
import os
import re
import ga
import ml
import aco
import json

# Function to check if a string is numeric
def is_numeric(input_string):
    try:
        float(input_string) # Attempt to convert the input string to a float
        return True  # Return True if successful
    except ValueError:
        return False  # Return False if ValueError is raised (not numeric)

# Function to check if a string is hexadecimal
def is_hexadecimal(s):
    pattern = r"^[0-9A-Fa-f]+$"  # Regular expression pattern for hexadecimal
    return bool(re.match(pattern, s))  # Return True if the string matches the pattern

# Function to calculate the average of a list of numbers
def calculate_list_average(lst):
    return sum(lst) / len(lst) if lst else 0  # Return the average or 0 if the list is empty

# Function to load data from a CSV file
def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]  # Read rows from the CSV file
    return data  # Return the loaded data as a list

# Function to fix trailing slash character in a string
def fix_trailing_character(input_string):
    return input_string.rstrip('/') + '/'  # Remove trailing '/' and add it back

# Function to remove duplicate rows from a list
def remove_duplicates_rows(input_list):
    return [list(t) for t in set(tuple(row) for row in input_list)]  # Use sets to remove duplicates

# Function to remove rows with all empty entries from CSV data
def remove_rows_with_nan(csv_data):
    return [list(row) for row in csv_data if not all(entry == "" for entry in row[:-1])]  # Filter rows with non-empty entries

# Function to modify a dataset represented as a list of lists (CSV data)
def modify_dataset(csv_data):
    # Iterate through rows of the dataset
    for i in range(len(csv_data)):
        # Iterate through columns of each row except the last one
        for j in range(len(csv_data[i]) - 1):
            cell = csv_data[i][j]

            # If the cell is empty, replace it with 'NaN'
            if len(cell) == 0:
                csv_data[i][j] = 'NaN'
            else:
                tokens = cell.split(',')
                total = 0

                # Loop through tokens in the cell
                for token in tokens:
                    if token.startswith("0x"):  # Check if the token is hexadecimal
                        # Convert hexadecimal to decimal
                        decimal_value = int(token, 16)
                        token = str(decimal_value)
                    elif not is_numeric(token.replace(',', '')):  # Check if the token is alphanumeric
                        # Calculate the hash value using the hash() function
                        decimal_hash = hash(token)
                        token = str(decimal_hash)
                    
                     # Sum up the numeric values after potential conversions
                    total += float(token)
                
                # Replace the cell value with the remainder modulo a large constant
                csv_data[i][j] = str(total % 0xFFFFFFFF)

# Function to read blacklisted features from a file
def read_blacklisted_features(blacklist_file_path):
    if os.path.exists(blacklist_file_path):
        with open(blacklist_file_path, 'r') as f:
            return f.read().splitlines()
    else:
        print(f"The file '{blacklist_file_path}' was not found.")
        sys.exit(1)

# Function to read feature names and filter blacklisted features
def read_and_filter_feature_names(feature_names_file_path, blacklisted_features):
    if os.path.exists(feature_names_file_path):
        with open(feature_names_file_path, 'r') as f:
            feature_names = f.read().splitlines()
            feature_names = [feature for feature in feature_names if feature not in blacklisted_features]
            feature_names.append('label')
        return feature_names
    else:
        print(f"The file '{feature_names_file_path}' was not found.")
        sys.exit(1)

# Function to determine and remove empty fields from CSV files
def remove_empty_fields_from_csv_files(csv_file_paths):
    intersection_of_columns_with_all_nan = {}
    for i in range(len(csv_file_paths)):
        df = pd.read_csv(csv_file_paths[i], low_memory=False)

        columns_with_all_nan = df.columns[df.isnull().all()].tolist()
        if len(intersection_of_columns_with_all_nan) == 0:
            intersection_of_columns_with_all_nan = set(columns_with_all_nan)
        else:
            intersection_of_columns_with_all_nan = intersection_of_columns_with_all_nan & set(columns_with_all_nan)

    if len(intersection_of_columns_with_all_nan) > 0:
        print("removing empty fields...")
        for i in range(len(csv_file_paths)):
            df = pd.read_csv(csv_file_paths[i], low_memory=False)
            df = df.drop(columns=intersection_of_columns_with_all_nan, errors='ignore')
            df.to_csv(csv_file_paths[i], index=False, na_rep='NaN')

# Function to write header row with feature names to CSV files
def write_header_to_csv_files(csv_file_paths, feature_names):
    for i in range(len(csv_file_paths)):
        with open(csv_file_paths[i], 'w') as f:
            csv_line = ','.join(feature_names)
            f.write(f"{csv_line}\n")

# Function to write selected field list to a file
def write_selected_field_list_to_file(csv_file_paths, selected_field_list_file_path):
    selected_field_list = []
    with open(csv_file_paths[0], 'r') as file:
        csv_reader = csv.reader(file)
        selected_field_list = next(csv_reader)

    with open(selected_field_list_file_path, 'w') as file:
        file.write(','.join(selected_field_list[:-1]))

# Write the packets to CSV file
def write_packets_to_csv_files(csv_file_paths, num_of_lines_per_file, csv_data):
    for i in range(len(csv_file_paths)):
        start_idx = i * num_of_lines_per_file
        end_idx = start_idx + num_of_lines_per_file if i < 2 else None
        
        file_data = csv_data[start_idx:end_idx]

        with open(csv_file_paths[i], 'a') as f:
            for line in file_data:
                csv_line = ','.join(line)
                f.write(f"{csv_line}\n")

def extract_features_from_pcap(blacklist_file_path, feature_names_file_path, protocol_folder_path, csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, selected_field_list_file_path):
    # Create protocol folder
    if not os.path.exists(protocol_folder_path):
        os.makedirs(protocol_folder_path)

    # Read blacklisted features
    blacklisted_features = read_blacklisted_features(blacklist_file_path)

    # Read feature names from the text file
    feature_names = read_and_filter_feature_names(feature_names_file_path, blacklisted_features)

    # Write header row with feature names
    write_header_to_csv_files(csv_file_paths, feature_names)

    # List of classes (dict)
    list_of_classes = {}
    list_counter = 0

    # Loop through each pcap file in the provided folder
    for index in range(len(pcap_file_names)):
        pcap_file_name = pcap_file_names[index]
        class_name = pcap_file_name.split('.pcap')[0]
        list_of_classes[list_counter] = class_name
        pcap_file_path = pcap_file_paths[index]

        print("processing " + pcap_file_name + "...")

        # Prepare the tshark command to be executed
        tshark_cmd = ['tshark', '-r', pcap_file_path, '-T', 'fields']
        for feature in feature_names[:-1]:
            tshark_cmd.append('-e')
            tshark_cmd.append(feature)

        # Call tshark to extract the specified features from the pcap file
        tshark_output = subprocess.check_output(tshark_cmd, universal_newlines=True)

        # Parse tshark output and write to CSV file
        csv_data = [line.split('\t') + [str(list_counter)] for line in tshark_output.strip().split('\n') if len(line.split('\t')) == len(feature_names) - 1]

        # Remove duplicates
        csv_data = remove_duplicates_rows(csv_data)

        # Filter out rows where the 'label' column is not 'NaN'
        csv_data = remove_rows_with_nan(csv_data)

        # Modify dataset
        modify_dataset(csv_data)

        # Calculate the number of lines per file
        num_of_lines_per_file = len(csv_data) // 3

        # Write the packets to CSV file
        write_packets_to_csv_files(csv_file_paths, num_of_lines_per_file, csv_data)

        list_counter += 1
    print()

    # Write class list to file
    with open(classes_file_path, 'w') as json_file:
        json.dump(list_of_classes, json_file)

    # Determine features that are empty across all 3 batches
    print("determining empty fields...")
    remove_empty_fields_from_csv_files(csv_file_paths)

    # Write field names to file
    write_selected_field_list_to_file(csv_file_paths, selected_field_list_file_path)

if __name__ == '__main__':
    # Check if at least one argument (excluding the script name) is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py arg1 arg2...")
        sys.exit(1)

    classifier_index = 0
    folder = ""
    protocol = ""
    pcap_folder_path = ""
    blacklist_file_path = ""
    feature_names_file_path = ""
    protocol_folder_path = ""
    classes_file_path = ""
    selected_field_list_file_path = ""
    log_file_path = "log.txt"
    csv_file_paths = []
    pcap_file_names = []
    pcap_file_paths = []
    fitness_function_file_paths = []
    selected_field_list = []
    num_of_iterations = 10
    num_of_packets_to_process = 0
    order_of_batches = [1,2,3]
    weights = [0.9,0.1]
    max_num_of_generations = 100
    filter_folder = os.path.join(os.path.dirname(__file__), "filters")
    mode = ""

    if not os.path.exists(filter_folder):
        print("The 'filters' folder is missing")
        sys.exit(1)

    # Loop through command-line arguments starting from the second element
    index = 1
    while index < len(sys.argv):
        if sys.argv[index] in ('-p', '--protocol'):
            if index + 1 < len(sys.argv):
                protocol = sys.argv[index + 1]
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -p/--protocol option")
                sys.exit(1)
        elif sys.argv[index] in ('-b', '--batch'):
            if index + 1 < len(sys.argv):
                order_of_batches = sys.argv[index + 1].split(',')
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -b/--batch option")
                sys.exit(1)
        elif sys.argv[index] in ('-w', '--weights'):
            if index + 1 < len(sys.argv):
                weights = [float(value) for value in sys.argv[index + 1].split(',')]
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -w/--weights option")
                sys.exit(1)
        elif sys.argv[index] in ('-n'):
            if index + 1 < len(sys.argv):
                num_of_packets_to_process = int(sys.argv[index + 1])
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -n option")
                sys.exit(1)
        elif sys.argv[index] in ('-f', '--folder'):
            if index + 1 < len(sys.argv):
                folder = fix_trailing_character(sys.argv[index + 1])
                pcap_folder = folder + "pcap"

                if not os.path.exists(folder):
                    print(f"The folder {folder} is missing")
                    sys.exit(1)
                if not os.path.exists(pcap_folder):
                    print("The 'pcap' folder is missing")
                    sys.exit(1)
                if len([f for f in os.listdir(pcap_folder) if f.endswith('.pcap')]) == 0:
                    print("There are no pcap files in the 'pcap' folder")
                    sys.exit(1)
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -f/--folder option")
                sys.exit(1)
        elif sys.argv[index] in ('-l', '--log'):
            if index + 1 < len(sys.argv):
                log_file_path = sys.argv[index + 1]

                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -f/--folder option")
                sys.exit(1)
        elif sys.argv[index] in ('-e', '--extract'):
            if folder == "" or protocol == "":
                print("Incorrect parameter order given!")
                sys.exit(1)

            # Set parameters
            pcap_folder_path = folder + "pcap"
            blacklist_file_path = f'{filter_folder}/blacklist.txt'
            feature_names_file_path = f'{filter_folder}/{protocol}.txt'
            protocol_folder_path = f'{folder}{protocol}'
            for i in range(3):
                csv_file_paths.append(f'{folder}{protocol}/batch_{i+1}.csv')
            pcap_file_names = [f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')]
            pcap_file_paths = [folder + "pcap/" + file_name for file_name in pcap_file_names]
            classes_file_path = f'{folder}/{protocol}/classes.json'
            selected_field_list_file_path = f'{folder}/{protocol}/fields.txt'

            mode = "extract"
            index += 1
        elif sys.argv[index] in ('-m', '--mode'):
            if folder == "":
                print("Incorrect parameter order given!")
                sys.exit(1)

            if index + 1 < len(sys.argv): # If there exists the value
                fitness_function_file_paths.append(f'{folder}{protocol}/batch_{order_of_batches[0]}.csv')
                fitness_function_file_paths.append(f'{folder}{protocol}/batch_{order_of_batches[1]}.csv')
                test_file_path = f'{folder}{protocol}/batch_{order_of_batches[2]}.csv'
                selected_field_list_file_path = f'{folder}/{protocol}/fields.txt'
                log_file_path = f'{folder}/{protocol}/{log_file_path}'
                classes_file_path = f'{folder}/{protocol}/classes.json'

                with open(selected_field_list_file_path, 'r') as file:
                    selected_field_list = file.readline().strip().split(',')

                mode = sys.argv[index+1]
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -m/--mode option")
                sys.exit(1)
        elif sys.argv[index] in ('-c', '--classifier'):
            if index + 1 < len(sys.argv):
                classifier_index = int(sys.argv[index+1])
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -c/--classifier option")
                sys.exit(1)
        else:
            print(f"Unknown parameter! '{sys.argv[index]}'")
            sys.exit(1)

    # Create the log file
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    with open(log_file_path, 'w') as file:
        pass

    # Run the mode
    if mode == 'extract':
        print("converting pcap files to csv format...\n")
        extract_features_from_pcap(blacklist_file_path, feature_names_file_path, protocol_folder_path, csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, selected_field_list_file_path)
    elif mode == 'ga':
        log("running GA...", log_file_path)
        best_solution, best_fitness = ga.run(fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations)
    elif mode == 'aco':
        log("running ACO...", log_file_path)
        best_solution, best_fitness = aco.run(fitness_function_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations)
    else:
        print("Unknown entry for the mode")

    # Print results
    if mode == 'ga' or mode == 'aco':
        log(f"Best Solution:\t[{''.join(map(str, best_solution))}]\tFitness: {best_fitness}", log_file_path)
        log("\nSelected features:", log_file_path)
        for i in range(len(best_solution)):
            if best_solution[i] == 1:
                log(selected_field_list[i], log_file_path)
        f1_score_average, predictions, test_labels = ml.classify_after_filtering(best_solution, fitness_function_file_paths, test_file_path, classifier_index)

        log("\nSelected feature-set accuracy: " + str(f1_score_average), log_file_path)

        # You can also print a classification report for more detailed metrics
        log("\nClassification Report:", log_file_path)
        log(classification_report(test_labels, predictions, zero_division=0), log_file_path)


    f1_score_average, predictions, test_labels = ml.classify_after_filtering(best_solution, fitness_function_file_paths, test_file_path, classifier_index, True)

    log("All feature-set accuracy: " + str(f1_score_average), log_file_path)

    # You can also print a classification report for more detailed metrics
    log("\nClassification Report:", log_file_path)
    log(classification_report(test_labels, predictions, zero_division=0), log_file_path)

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
import statistics

def is_numeric(input_string):
    try:
        float(input_string) # Attempt to convert the input string to a float
        return True  # Return True if successful
    except ValueError:
        return False  # Return False if ValueError is raised (not numeric)

def is_hexadecimal(s):
    return bool(re.match(r"^[0-9A-Fa-f]+$", s))

def calculate_list_average(lst):
    return sum(lst) / len(lst) if lst else 0  # Return the average or 0 if the list is empty

def fix_trailing_character(input_string):
    return input_string.rstrip('/') + '/'  # Remove trailing '/' and add it back

def remove_duplicates_rows(csv_data):
    seen = set()
    unique_csv_data = []

    for sublist in csv_data:
        tuple_sublist = tuple(sublist)

        if tuple_sublist not in seen:
            unique_csv_data.append(sublist)
            seen.add(tuple_sublist)

    return unique_csv_data

def remove_rows_with_nan_values(csv_data):
    return [list(row) for row in csv_data if not all(entry == "" for entry in row[:-1])]  # Filter rows with non-empty entries

def modify_dataset(csv_data):
    # Iterate through rows of the dataset
    for i in range(len(csv_data)):
        # Iterate through columns of each row except the label
        for j in range(len(csv_data[i]) - 1):
            cell = csv_data[i][j]

            # If the cell is empty, replace it with '-1'
            if len(cell) == 0:
                csv_data[i][j] = '-1'
            else:
                tokens = cell.split(',')
                total = 0

                # Loop through tokens in the cell
                for token in tokens:
                    if token.startswith("0x"):  # Check if the token is hexadecimal
                        # Convert hexadecimal to decimal
                        decimal_value = int(token, 16)
                        token = str(decimal_value)
                    elif not is_numeric(token):  # Check if the token is alphanumeric
                        # Calculate the hash value using the hash() function
                        decimal_hash = hash(token)
                        token = str(decimal_hash)
                    
                     # Sum up the numeric values after potential conversions
                    total += float(token)
                
                # Replace the cell value with the remainder modulo a large constant
                csv_data[i][j] = str(total % 0xFFFFFFFF)

def read_blacklisted_features(blacklist_file_path):
    try:
        with open(blacklist_file_path, 'r') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"The file '{blacklist_file_path}' was not found.")
        sys.exit(1)

def read_and_filter_feature_names(feature_names_file_path, blacklisted_features):
    try:
        with open(feature_names_file_path, 'r') as f:
            feature_names = [feature for feature in f.read().splitlines() if feature not in blacklisted_features]
            feature_names.append('label')
        return feature_names
    except FileNotFoundError:
        print(f"The file '{feature_names_file_path}' was not found.")
        sys.exit(1)

def add_stat_features_to_csv_files(csv_file_paths):
    for file_path in csv_file_paths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Create a list to store the new header
        new_header = []

        # Iterate through the existing header and create new columns for statistics
        for field in df.columns[:-1]:  # Exclude the last column (label)
            new_header.extend([field])
        for field in df.columns[:-1]:  # Exclude the last column (label)
            new_header.extend([f"{field}_min", f"{field}_max", f"{field}_mean", f"{field}_std", f"{field}_mode"])
        new_header.append(df.columns[-1])

        # Calculate statistics for each numeric column
        stats = []
        for field in df.columns[:-1]:  # Exclude the last column (label)
            column_values = df[field].tolist()
            stats.extend([min(column_values), max(column_values), statistics.mean(column_values), statistics.stdev(column_values), statistics.mode(column_values)])

        csv_data = []
        csv_data.append(new_header)

        # Iterate through the DataFrame rows and append the list
        for index, row in df.iterrows():
            if index == 0:
                continue  # Skip the header row
            row_list = row.tolist()
            row_list = row_list[:-1] + stats + [int(row_list[-1])]
            csv_data.append(row_list)

        with open(file_path, 'w') as file:
            for inner_list in csv_data:
                line = ','.join(map(str, inner_list))  # Convert inner list to a comma-separated string
                file.write(line + '\n')

def remove_empty_fields_from_csv_files(csv_file_paths):
    # Read all CSV files into a list of DataFrames
    dfs = [pd.read_csv(file_path, low_memory=False) for file_path in csv_file_paths]

    # Find columns with only one unique value across all DataFrames
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns &= set(df.columns)

    columns_to_remove = set()

    # Check if each common column has only one unique value across all DataFrames
    for col in common_columns:
        common_values = set(dfs[0][col].unique())
        for df in dfs[1:]:
            common_values &= set(df[col].unique())
        if len(common_values) == 1:
            columns_to_remove.add(col)

    if columns_to_remove:
        print("removing empty fields...")
        for i in range(len(csv_file_paths)):
            df = pd.read_csv(csv_file_paths[i], low_memory=False)
            df = df.drop(columns=columns_to_remove, errors='ignore')
            df.to_csv(csv_file_paths[i], index=False)

def write_header_to_csv_files(csv_file_paths, feature_names):
    for i in range(len(csv_file_paths)):
        with open(csv_file_paths[i], 'w') as f:
            csv_line = ','.join(feature_names)
            f.write(f"{csv_line}\n")

def write_remaining_field_list_to_file(csv_file_paths, selected_field_list_file_path):
    selected_field_list = []
    with open(csv_file_paths[0], 'r') as file:
        csv_reader = csv.reader(file)
        selected_field_list = next(csv_reader)

    with open(selected_field_list_file_path, 'w') as file:
        file.write(','.join(selected_field_list[:-1]))

def write_packets_to_csv_files(csv_file_paths, num_of_lines_per_file, csv_data):
    for i in range(len(csv_file_paths)):
        start_idx = i * num_of_lines_per_file
        end_idx = start_idx + num_of_lines_per_file if i < 2 else None
        
        file_data = csv_data[start_idx:end_idx]

        with open(csv_file_paths[i], 'a') as f:
            for line in file_data:
                csv_line = ','.join(line)
                f.write(f"{csv_line}\n")

def extract_features_from_pcap(blacklist_file_path, feature_names_file_path, protocol_folder_path, csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, selected_field_list_file_path, statistical_features_on):
    # Create protocol folder
    if not os.path.exists(protocol_folder_path):
        os.makedirs(protocol_folder_path)

    # Read blacklisted features
    blacklisted_features = read_blacklisted_features(blacklist_file_path)

    # Read feature names from the filters folder
    csv_header = read_and_filter_feature_names(feature_names_file_path, blacklisted_features)

    # Write header row with feature names to csv files
    write_header_to_csv_files(csv_file_paths, csv_header)

    # List of classes (dict)
    list_of_classes = {}
    class_counter = 0

    # Loop through each pcap file in the provided folder
    for index in range(len(pcap_file_names)):
        pcap_file_name = pcap_file_names[index]
        class_name = pcap_file_name.split('.pcap')[0]
        list_of_classes[class_counter] = class_name
        pcap_file_path = pcap_file_paths[index]

        print("processing " + pcap_file_name + "...")

        # Prepare the tshark command to be executed
        tshark_cmd = ['tshark', '-r', pcap_file_path, '-T', 'fields']
        for feature in csv_header[:-1]:
            tshark_cmd.append('-e')
            tshark_cmd.append(feature)

        # Call tshark to extract the specified features from the pcap file
        tshark_output = subprocess.check_output(tshark_cmd, universal_newlines=True)

        # Parse tshark output and write to CSV file
        csv_data = [line.split('\t') + [str(class_counter)] for line in tshark_output.strip().split('\n') if len(line.split('\t')) == len(csv_header) - 1]

        del tshark_output

        # Filter out rows where the 'label' column is not 'NaN'
        csv_data = remove_rows_with_nan_values(csv_data)

        # Remove duplicates
        csv_data = remove_duplicates_rows(csv_data)

        # Modify dataset
        modify_dataset(csv_data)

        # Calculate the number of lines per file
        num_of_lines_per_file = len(csv_data) // 3

        # Write the packets to CSV file
        write_packets_to_csv_files(csv_file_paths, num_of_lines_per_file, csv_data)

        class_counter += 1
    print()

    # Write class list to file
    with open(classes_file_path, 'w') as json_file:
        json.dump(list_of_classes, json_file)

    # Determine features that are empty across all 3 batches
    print("determining empty fields...")
    remove_empty_fields_from_csv_files(csv_file_paths)

    if statistical_features_on:
        add_stat_features_to_csv_files(csv_file_paths)

    # Write remaining field names to file
    write_remaining_field_list_to_file(csv_file_paths, selected_field_list_file_path)

def print_usage():
    """Print a usage guide for the program."""
    print("Usage: python autoIC.py [options]")
    print("Options:")
    print("  -p, --protocol    Specify the protocol (e.g., http, ftp)")
    print("  -b, --batch       Specify a comma-separated list of batch orders")
    print("  -i, --iteration   Specify the number of iterations")
    print("  -g, --generation  Specify the maximum number of generations")
    print("  -w, --weights     Specify weights as a comma-separated list")
    print("  -n                Specify the number of packets to process")
    print("  -f, --folder      Specify the folder path")
    print("  -l, --log         Specify the log file path")
    print("  -e, --extract     Set the mode to 'extract'")
    print("  -m, --mode        Specify the mode (e.g., 'mode1', 'mode2')")
    print("  -c, --classifier  Specify the classifier index (integer)")
    print("  -h, --help        Display this help message")
    print()
    print("Example:")
    print("python autoIC.py -p http -b batch1,batch2 -i 100 -g 10 -w 0.5,0.3 -n 50 -f /path/to/folder -l logfile.txt -e -m mode1 -c 0")
    print()
    print("Note:")
    print("- Use commas to separate multiple values for options that accept lists (e.g., -b, -w).")
    print("- When using options with values, ensure that the value follows the option without spaces.")
    print("- Use -h or --help to display this help message.")

if __name__ == '__main__':
    if len(sys.argv) < 2: # check if at least one argument is provided
        print("Usage: python script.py arg1 arg2...")
        sys.exit(1)

    # Determine filters folder path
    filters_folder = os.path.join(os.path.dirname(__file__), "filters")
    if not os.path.exists(filters_folder):
        print("The 'filters' folder is missing")
        sys.exit(1)

    # Variables
    classifier_index = 0
    max_num_of_generations = 50
    num_of_iterations = 10
    num_of_packets_to_process = 0
    order_of_batches = [1,2,3]
    weights = [0.9,0.1]
    log_file_path = "log.txt"
    mode = ""
    folder = ""
    protocol = ""
    statistical_features_on = False

    # Loop through command-line arguments starting from the second element
    index = 1
    while index < len(sys.argv):
        if sys.argv[index] in ('-p', '--protocol'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -p/--protocol option")
                sys.exit(1)

            protocol = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-b', '--batch'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -b/--batch option")
                sys.exit(1)
            
            order_of_batches = sys.argv[index + 1].split(',')
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-i', '--iteration'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -i/--iteration option")
                sys.exit(1)
            
            num_of_iterations = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-g', '--generation'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -g/--generation option")
                sys.exit(1)
            
            max_num_of_generations = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-w', '--weights'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -w/--weights option")
                sys.exit(1)

            weights = [float(value) for value in sys.argv[index + 1].split(',')]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-n'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -n option")
                sys.exit(1)

            num_of_packets_to_process = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-f', '--folder'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -f/--folder option")
                sys.exit(1)

            folder = fix_trailing_character(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-l', '--log'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -f/--folder option")
                sys.exit(1)

            log_file_path = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-e', '--extract'):
            mode = "extract"
            index += 1
        elif sys.argv[index] in ('-s', '--statistics'):
            statistical_features_on = True
            index += 1
        elif sys.argv[index] in ('-m', '--mode'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -m/--mode option")
                sys.exit(1)

            mode = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-h', '--help'):
            print_usage()
            sys.exit(0)
        elif sys.argv[index] in ('-c', '--classifier'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -c/--classifier option")
                sys.exit(1)

            classifier_index = int(sys.argv[index+1])
            index += 2  # Skip both the option and its value
        else:
            print(f"Unknown parameter! '{sys.argv[index]}'")
            sys.exit(1)

    # Set parameters and perform validation checks
    if folder == "":
        print("Workspace folder not given!")
        sys.exit(1)

    if not os.path.exists(folder):
        print(f"The folder {folder} is missing")
        sys.exit(1)

    pcap_folder_path = folder + "pcap"
    if not os.path.exists(pcap_folder_path):
        print("The 'pcap' folder is missing")
        sys.exit(1)

    if len([f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')]) == 0:
        print("There are no pcap files in the 'pcap' folder")
        sys.exit(1)

    log_file_path = f'{folder}/{protocol}/{log_file_path}'
    if os.path.exists(log_file_path):
        os.remove(log_file_path) # Remove previous log file

    csv_file_paths = []
    classes_file_path = f'{folder}/{protocol}/classes.json'
    train_file_paths = []
    selected_field_list_file_path = f'{folder}/{protocol}/fields.txt'
    selected_field_list = []
    if os.path.exists(selected_field_list_file_path):
        with open(selected_field_list_file_path, 'r') as file:
            selected_field_list = file.readline().strip().split(',')

    # Run the mode
    if mode == 'extract':
        blacklist_file_path = f'{filters_folder}/blacklist.txt'
        feature_names_file_path = f'{filters_folder}/{protocol}.txt'
        protocol_folder_path = f'{folder}{protocol}'
        for i in range(3):
            csv_file_paths.append(f'{folder}{protocol}/batch_{i+1}.csv')
        pcap_file_names = [f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')]
        pcap_file_paths = [folder + "pcap/" + file_name for file_name in pcap_file_names]

        print("converting pcap files to csv format...\n")
        extract_features_from_pcap(
            blacklist_file_path, feature_names_file_path, protocol_folder_path,
            csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path,
            selected_field_list_file_path, statistical_features_on
        )
    elif mode == 'ga' or mode == 'aco':
        train_file_paths.append(f'{folder}{protocol}/batch_{order_of_batches[0]}.csv')
        train_file_paths.append(f'{folder}{protocol}/batch_{order_of_batches[1]}.csv')
        test_file_path = f'{folder}{protocol}/batch_{order_of_batches[2]}.csv'
        fields_file_path = f'{folder}{protocol}/fields.txt'

        if mode == 'ga':
            log("running GA...\n", log_file_path)
            best_solution, best_fitness = ga.run(
                train_file_paths, classifier_index, classes_file_path,
                num_of_packets_to_process, num_of_iterations, weights,
                log_file_path, max_num_of_generations, fields_file_path
            )
        elif mode == 'aco':
            log("running ACO...\n", log_file_path)
            best_solution, best_fitness = aco.run(
                train_file_paths, classifier_index, classes_file_path,
                num_of_packets_to_process, num_of_iterations, weights,
                log_file_path, max_num_of_generations, fields_file_path
            )

        # Print best solution and the features selected
        sol_str = ''.join(map(str, best_solution))
        log(f"Best Solution:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}", log_file_path)
        log("\nSelected features:", log_file_path)
        for i in range(len(best_solution)):
            if best_solution[i] == 1:
                log(selected_field_list[i], log_file_path)

        # Print the classification result on test data using selected features
        log("", log_file_path)
        log("Selected feature-set results:", log_file_path)
        ml.classify_after_filtering(best_solution, train_file_paths, test_file_path, classifier_index, log_file_path, True)
        
        # Print the classification result on test data using all features
        log("All feature-set results:", log_file_path)
        ml.classify_after_filtering(best_solution, train_file_paths, test_file_path, classifier_index, log_file_path, False)
    else:
        print("Unknown entry for the mode")

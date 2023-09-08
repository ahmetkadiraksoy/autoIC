import subprocess
import csv
import sys
import os
import re
import ga
import aco
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

MAX_32BIT_INT = 2 ** 32

def is_numeric(input_string):
    try:
        float(input_string)  # Try to convert the string to a float
        return True
    except ValueError:
        return False

def is_hexadecimal(s):
    # Define a regular expression pattern for a hexadecimal string
    pattern = r"^[0-9A-Fa-f]+$"

    # Use re.match() to check if the string matches the pattern
    if re.match(pattern, s):
        return True
    else:
        return False

def calculate_list_average(lst):
    return sum(lst) / len(lst) if lst else 0

# Load and prepare data
def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def fix_trailing_character(input_string):
    if not input_string.endswith('/'):
        return input_string + '/'
    return input_string

def remove_duplicates_list_list(input_list):
    return [list(t) for t in set(tuple(row) for row in input_list)]

def extract_features_from_pcap(pcap_folder_path, protocol, blacklisted_features):
    # Create protocol folder
    folder_path = f'{pcap_folder_path}/{protocol}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Paths to input pcap file, feature names text file, and output CSV file
    feature_names_file = f'{pcap_folder_path}{protocol}.txt'

    file_list = [f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')]

    # Read feature names from the text file
    if os.path.exists(feature_names_file):
        with open(feature_names_file, 'r') as f:
            feature_names = f.read().splitlines()
            feature_names = [feature for feature in feature_names if feature not in blacklisted_features]
    else:
        print(f"The file '{feature_names_file}' was not found.")
        sys.exit(1)

    # Write header row with feature names
    for i in range(3):
        with open(f'{pcap_folder_path}/{protocol}/batch_{i+1}.csv', 'w') as f:
            csv_line = ','.join(feature_names)
            f.write(f"{csv_line},label\n")

    # Loop through each pcap file in the provided folder
    for file in file_list:
        print("processing " + file + "...")
        class_name = file.split('.pcap')[0]
        pcap_file_path = pcap_folder_path + file

        # Prepare the tshark command to be executed
        tshark_cmd = ['tshark', '-r', pcap_file_path, '-T', 'fields']
        for feature in feature_names:
            tshark_cmd.append('-e')
            tshark_cmd.append(feature)

        # Call tshark to extract the specified features from the pcap file
        tshark_output = subprocess.check_output(tshark_cmd, universal_newlines=True)

        # Parse tshark output and write to CSV file
        csv_data = [line.split('\t') + [class_name] for line in tshark_output.strip().split('\n')]

        # Remove duplicates
        csv_data = remove_duplicates_list_list(csv_data)

        # Modify dataset
        for i in range(len(csv_data)):
            for j in range(len(csv_data[i]) - 1):
                if len(csv_data[i][j]) == 0:
                    csv_data[i][j] = 'NaN'
                else:
                    cell = csv_data[i][j]
                    tokens = cell.split(',')

                    total = 0
                    for token in tokens:
                        if token.startswith("0x"): # Hexadecimal
                            token = str(int(token[2:], 16))
                        elif not is_numeric(cell.replace(',', '')): # Alphanumeric
                            # Calculate the hash value using the hash() function
                            decimal_hash = hash(token)

                            # Ensure the hash value is non-negative (hash() can return negative values)
                            if decimal_hash < 0:
                                decimal_hash += MAX_32BIT_INT  # Convert to a non-negative 32-bit integer

                            token = decimal_hash
                        total += int(token)
                    csv_data[i][j] = str(total)

        # Calculate the number of lines per file
        lines_per_file = len(csv_data) // 3

        for i in range(3):
            start_idx = i * lines_per_file
            end_idx = start_idx + lines_per_file if i < 2 else None
            
            file_data = csv_data[start_idx:end_idx]
            
            with open(f'{pcap_folder_path}/{protocol}/batch_{i+1}.csv', 'a') as f:
                for line in file_data:
                    csv_line = ','.join(line)
                    f.write(f"{csv_line}\n")

    # Determine features that are empty across all 3 batches
    intersection_of_columns_with_all_nan = {}
    for i in range(3):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(f'{pcap_folder_path}/{protocol}/batch_{i+1}.csv')

        # Get a list of column names with all NaN values
        columns_with_all_nan = df.columns[df.isnull().all()].tolist()
        if len(intersection_of_columns_with_all_nan) == 0:
            intersection_of_columns_with_all_nan = set(columns_with_all_nan)
        else:
            intersection_of_columns_with_all_nan = intersection_of_columns_with_all_nan & set(columns_with_all_nan)

    # Remove the empty features from the csv files
    if len(intersection_of_columns_with_all_nan) > 0:
        for i in range(3):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(f'{pcap_folder_path}/{protocol}/batch_{i+1}.csv')

            # Remove the specified columns from the DataFrame
            df = df.drop(columns=intersection_of_columns_with_all_nan, errors='ignore')

            # Save the modified DataFrame back to a CSV file
            df.to_csv(f'{pcap_folder_path}/{protocol}/batch_{i+1}.csv', index=False, na_rep='NaN')

def main():
    # Check if at least one argument (excluding the script name) is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py arg1 arg2...")
        sys.exit(1)

    # List of classifiers to test
    classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    ]

    classifier_index = 0
    folder = ""
    protocol = ""
    blacklisted_features = [
        'ip.src',
        'ip.dst',
        'ip.id',
        'ip.geoip.src_isp',
        'ip.geoip.isp',
        'ip.geoip.dst_isp',
        'ip.dsfield.ect',
        'ip.dsfield.ce',
        'ip.checksum_good',
        'ip.checksum_bad',
        'ip.checksum',
        'ip.checksum.status',
        'ip.checksum_bad',
        'ip.checksum_bad.expert',
        'ip.checksum_calculated',
        'ip.checksum_good',
        'ip.dst_host',
        'ip.host',
        'ip.addr',
        'ip.src_host'
    ]

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
        elif sys.argv[index] in ('-f', '--folder'):
            if index + 1 < len(sys.argv):
                folder = fix_trailing_character(sys.argv[index + 1])
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -f/--folder option")
                sys.exit(1)
        elif sys.argv[index] in ('-e', '--extract'):
            if folder == "":
                print("Incorrect parameter order given!")
                sys.exit(1)

            print("converting pcap files to csv format...")
            extract_features_from_pcap(folder, protocol, blacklisted_features)
            index += 1
        elif sys.argv[index] in ('-m', '--mode'):
            if folder == "":
                print("Incorrect parameter order given!")
                sys.exit(1)

            if index + 1 < len(sys.argv):
                if sys.argv[index+1] == 'ga':
                    print("running GA...")
                    best_solution, best_fitness = ga.run(folder + '/' + protocol + '/batch_1.csv', folder + '/' + protocol + '/batch_2.csv', classifiers[classifier_index])
                    print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
                elif sys.argv[index+1] == 'aco':
                    print("running ACO...")
                    best_solution, best_fitness = aco.run(folder + '/' + protocol + '/batch_1.csv', folder + '/' + protocol + '/batch_2.csv', classifiers[classifier_index])
                    print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
                else:
                    print("Unknown entry for the mode")
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -m/--mode option")
                sys.exit(1)
        elif sys.argv[index] in ('-c', '--classify'):
            if index + 1 < len(sys.argv):
                classifier_index = int(sys.argv[index + 1])
                index += 2  # Skip both the option and its value
            else:
                print("Missing value for -f/--folder option")
                sys.exit(1)

        else:
            print(f"Unknown parameter! '{sys.argv[index]}'")
            sys.exit(1)

main()

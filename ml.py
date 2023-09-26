from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from libraries import log
from sklearn.metrics import classification_report
import csv

def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def remove_duplicates_list_list(list_of_lists):
    # Create an empty set to track unique sublists
    unique_sublists = set()

    # Create a new list to store the result with duplicates removed
    result = []

    # Iterate through the list of lists
    for sublist in list_of_lists:
        # Convert the sublist to a tuple to make it hashable
        sublist_tuple = tuple(sublist)
        
        # Check if the sublist is unique
        if sublist_tuple not in unique_sublists:
            unique_sublists.add(sublist_tuple)  # Add it to the set of unique sublists
            result.append(sublist)  # Append it to the result list

    # Now, 'result' contains the unique sublists with the first occurrence in place
    return result

def train_and_evaluate_classifier(classifier_index, train_features, train_labels, test_features, test_labels):
    # List of classifiers to test
    classifiers = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        SVC(random_state=42),
        LinearSVC(random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    ]

    clf = classifiers[classifier_index]
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    return f1_score(test_labels, predictions, average='weighted'), predictions, test_labels

def extract_features_and_labels(data, label_column_index):
    features = [row[:label_column_index] + row[label_column_index+1:] for row in data]
    labels = [row[label_column_index] for row in data]
    return features, labels

def classify(train, test, classifier_index):
    try:
        # Extract header and data separately
        train_header, train_data = train[0], train[1:]
        test_data = test[1:]

        # Find the index of the label column in the header
        label_column_index = train_header.index('label')

        # Extract features and labels
        train_features, train_labels = extract_features_and_labels(train_data, label_column_index)
        test_features, test_labels = extract_features_and_labels(test_data, label_column_index)

        # Train and evaluate the classifier
        return train_and_evaluate_classifier(classifier_index, train_features, train_labels, test_features, test_labels)
    except ValueError as e:
        print(f"Error: {e}")

def classify_after_filtering(solution, fitness_function_file_paths, test_file_path, classifier_index, log_file_path, filter):
    train = []
    train.extend(load_csv(fitness_function_file_paths[0]))
    train.extend(load_csv(fitness_function_file_paths[1])[1:]) # remove header
    test = load_csv(test_file_path)
    train = remove_duplicates_list_list(train)
    test = remove_duplicates_list_list(test)

    # Filter features
    if (filter):
        # Append 1 to the end so that it doesn't filter out the 'class' column
        solution_new = list(solution)
        solution_new.append(1)

        train = [[col for col, m in zip(row, solution_new) if m] for row in train]
        test = [[col for col, m in zip(row, solution_new) if m] for row in test]

    f1_score_average, predictions, test_labels = classify(train, test, classifier_index)

    log("\nAccuracy: " + str(f1_score_average), log_file_path)
    log("\nClassification Report:", log_file_path)
    log(classification_report(test_labels, predictions, zero_division=0), log_file_path)

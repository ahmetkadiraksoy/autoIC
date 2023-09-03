from sklearn.metrics import f1_score

def train_and_evaluate_classifier(clf, train_features, train_labels, test_features, test_labels):
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    return f1_score(test_labels, predictions, average='weighted')

def extract_features_and_labels(data, label_column_index):
    features = [row[:label_column_index] + row[label_column_index+1:-1] for row in data]
    labels = [row[label_column_index] for row in data]
    return features, labels

def classify(train, test, clf):
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
        return train_and_evaluate_classifier(clf, train_features, train_labels, test_features, test_labels)

    except ValueError as e:
        print(f"Error: {e}")

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from itertools import takewhile

def process_accuracies(batches_data, algorithms, feature_key):
    err_l, main_acc, err_u = [], [], []
    for algorithm in algorithms:
        acc = []
        for i in range(3):
            acc.append([batch_data[feature_key] for batch_data in batches_data if batch_data['classifier'] == algorithm and batch_data['batch_number'] == (i+1)][0])
        acc = sorted(acc)        
        err_l.append((acc[1] - acc[0])*100)
        main_acc.append(acc[1]*100)
        err_u.append((acc[2] - acc[1])*100)
    return err_l, main_acc, err_u

def plot(batches_data, classifiers, directory):
    ga_data = process_accuracies(batches_data, classifiers, 'selected_features_f1')
    nonga_data = process_accuracies(batches_data, classifiers, 'all_features_f1')

    fig, ax = plt.subplots(figsize=(3,5))
    ind, width = np.arange(len(classifiers)), 0.20

    for offset, color, data in zip([0.15, 0.15 + width], ['#1C6CAB', '#FF7311'], [ga_data, nonga_data]):
        ax.errorbar(ind + offset, data[1], yerr=[data[0], data[2]], mec=color, ecolor=color, fmt='o', capsize=4, mew=5)

    ax.set_xticks(ind + 0.10 + width / 2)
    ax.set_xticklabels(('J48', 'PART', 'DT', 'ANN'), fontsize=15)
    plt.subplots_adjust(left=0.14, right=0.99, top=0.94, bottom=0.05)
    plt.ylim([0, 100])
    plt.yticks(fontsize=15)
    plt.savefig(f"{directory}/plot.eps", format='eps', dpi=1000)

if __name__ == '__main__':
    directory = sys.argv[1]

    file_names = sorted(os.listdir(directory), key=lambda s: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)])
    full_paths = [os.path.join(directory, file) for file in file_names if "packets_" in file and "clf_" in file and file.endswith(".txt")]

    batches_data = []
    for path in full_paths:
        with open(path, 'r') as f:
            content = f.readlines()
        
        # Process file data
        validation_f1 = float(next(line.split()[-1] for line in content if "Best Solution:" in line))
        selected_features = [line.strip() for line in takewhile(lambda x: x.strip(), content[next(i for i, line in enumerate(content) if "Selected features:" in line) + 1:])]
        f1_values = [float(line.split()[-1]) for line in content if "F1" in line]
        batches_data.append({
            "mode": path.split('_')[3],
            "classifier": int(path.split('_')[5]),
            "batch_number": int(path.split('_')[7]),
            "run_number": int(path.split('_')[9].split(".")[0]),
            "selected_features": selected_features,
            "validation_f1": validation_f1,
            "selected_features_f1": f1_values[-2],
            "all_features_f1": f1_values[-1],
            "file_path": path
        })

    for batch_data in batches_data:
        print(batch_data['file_path'])
    classifiers = sorted(list(set(batch_data['classifier'] for batch_data in batches_data)))
    plot(batches_data, classifiers, directory)

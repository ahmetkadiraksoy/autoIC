import os
import re
import numpy as np
import matplotlib.pyplot as plt
from itertools import takewhile

def process_accuracies(batches_data, clfs, feature_key):
    err_l, main_acc, err_u = [], [], []
    for clf in clfs:
        acc = []
        for i in range(3):
            acc.append([batch_data[feature_key] for batch_data in batches_data if batch_data['classifier'] == clf and batch_data['batch_number'] == (i+1)][0])
        acc = sorted(acc)        
        err_l.append((acc[1] - acc[0])*100)
        main_acc.append(acc[1]*100)
        err_u.append((acc[2] - acc[1])*100)
    return err_l, main_acc, err_u

def plot(batches_data, clfs, folder, classifiers, mode):
    ga_data = process_accuracies(batches_data, clfs, 'selected_features_f1')
    nonga_data = process_accuracies(batches_data, clfs, 'all_features_f1')
    labels = [classifiers[int(clf)][0] for clf in clfs]

    ax = plt.subplots(figsize=(3,5))[1]
    ind, width = np.arange(len(clfs)), 0.30

    for offset, color, data in zip([0.15, 0.15 + width], ['#1C6CAB', '#FF7311'], [ga_data, nonga_data]):
        ax.errorbar(ind + offset, data[1], yerr=[data[0], data[2]], mec=color, ecolor=color, fmt='o', capsize=4, mew=5)

    ax.set_xticks(ind + 0.10 + width / 2)
    ax.set_xticklabels(labels, fontsize=15)
    plt.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.06)
    plt.ylim([0, 100])
    plt.yticks(fontsize=15)
    plt.savefig(f"{folder}/plot_" + mode + ".eps", format='eps', dpi=1000)

def run(folder, classifiers):
    file_names = sorted(os.listdir(folder), key=lambda s: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)])
    full_paths = [os.path.join(folder, file) for file in file_names if file.startswith("packets_") and file.endswith(".txt")]

    modes = list(set([os.path.basename(path).split('_')[3] for path in full_paths]))
    for mode in modes:
        batches_data = []
        for path in full_paths:
            filename = os.path.basename(path)
            if filename.split('_')[3] == mode:
                with open(path, 'r') as f:
                    content = f.readlines()
                
                # Process file data
                validation_f1 = float(next(line.split()[-1] for line in content if "Best Solution:" in line))
                selected_features = [line.strip() for line in takewhile(lambda x: x.strip(), content[next(i for i, line in enumerate(content) if "Selected features:" in line) + 1:])]
                f1_values = [float(line.split()[-1]) for line in content if "F1" in line]
                batches_data.append({
                    "mode": filename.split('_')[3],
                    "classifier": int(filename.split('_')[5]),
                    "batch_number": int(filename.split('_')[7]),
                    "run_number": int(filename.split('_')[9].split(".")[0]),
                    "selected_features": selected_features,
                    "validation_f1": validation_f1,
                    "selected_features_f1": f1_values[-2],
                    "all_features_f1": f1_values[-1],
                    "file_path": path
                })

        clfs = sorted(list(set(batch_data['classifier'] for batch_data in batches_data)))
        plot(batches_data, clfs, folder, classifiers, mode)

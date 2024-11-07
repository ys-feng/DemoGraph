import pickle
import os
import pandas as pd
import numpy as np
import dgl
import torch


def load_first_n_files(directory, n):
    data_frames = []
    total_errors = 0
    count = 0

    for filename in os.listdir(directory):
        if count >= n:
            break
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path, delimiter='\t', header=None)
                data_frames.append(df)
            except pd.errors.ParserError as e:
                total_errors += 1
                print(f"Error in file: {file_path}")
                line_number = int(str(e).split("line ")[-1].split(",")[0])
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                lines.pop(line_number - 1)
                with open(file_path, 'w') as f:
                    f.writelines(lines)
        count += 1

    print(f"Total parsing errors: {total_errors}")
    return data_frames


def map_labels_to_integers(labels):
    unique_labels, indices = np.unique(labels, return_inverse=True)
    return indices


def map_df_to_integers(df, mapping):
    indices = mapping[df.values.ravel()].reshape(df.shape)
    return indices


# Load data
cond_data = load_first_n_files('./graphs/cond', 50)
proc_data = load_first_n_files('./graphs/proc', 50)
drug_data = load_first_n_files('./graphs/drug', 50)

# Load mimic3_dp data
with open('/home/r10user16/GraphAug/data/graphs/mimic3_dp.pkl', 'rb') as f:
    mimic3_dp_data = pickle.load(f)

# Extract nodes and edges from mimic3_dp_data
mimic3_nodes = mimic3_dp_data.nodes
mimic3_edges = mimic3_dp_data.edges

# mimic3_nodes_df = pd.DataFrame(mimic3_nodes)
# mimic3_edges_df = pd.DataFrame(mimic3_edges)

# Combine all data frames
all_data = cond_data + proc_data + drug_data + [mimic3_dp_data]

# Create a mapping for all labels
all_labels = np.concatenate([df.values.flatten() for df in all_data])
mapping = map_labels_to_integers(all_labels)

# Create a heterograph with integer node labels
graph_data = {}

# Create and save knowledge graphs for each data
for i, df in enumerate(all_data):
    try:
        if isinstance(df, pd.DataFrame):
            mapped_df = map_df_to_integers(df, mapping)
            heads, tails = mapped_df[:, 0], mapped_df[:, 2]
        else:
            #?
            pass
    except IndexError:
        continue

    graph_data[('node', 'related_to', 'node')] = (heads, tails)

    g = dgl.heterograph(graph_data)

    # Connect all graphs through a common node
    if i > 0:
        g = dgl.compose(g_prev, g)

    g_prev = g

# Save the combined graph
with open('combined_graph.pkl', 'wb') as f:
    pickle.dump(g, f)
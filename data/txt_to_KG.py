import pickle
import os
import pandas as pd
import numpy as np
import dgl
import torch

# Function to load data from all txt files in a directory into a list of DataFrames
def load_data_from_directory(directory):
    data_frames = []
    total_errors = 0  # Counter for total parsing errors

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                try:
                    df = pd.read_csv(file, delimiter='\t', header=None)  # Set header=None for DataFrames without column names
                    data_frames.append(df)
                except pd.errors.ParserError as e:
                    total_errors += 1
                    print(f"Error in file: {file_path}")
                    # Get the line number from the error message
                    line_number = int(str(e).split("line ")[-1].split(",")[0])
                    # Read all the lines from the file
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    # Remove the problematic line from the list
                    lines.pop(line_number - 1)
                    # Write the cleaned lines back to the file
                    with open(file_path, 'w') as f:
                        f.writelines(lines)

    print(f"Total parsing errors: {total_errors}")
    return data_frames

# Function to map each node or edge label to an integer
def map_labels_to_integers(labels):
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_to_int[label] for label in labels])

# Function to map labels to integers using the mappings
def map_labels_to_integers_df_h(df, mapping):
    entity_set = set(df.iloc[:, 0].values)
    mp = {e: i for i, e in enumerate(entity_set)}
    indices = [mp[v] for v in df.iloc[:, 0].values]
    return indices
    # return torch.tensor(mapping[df.iloc[:, 0].values])

def map_labels_to_integers_df_t(df, mapping):
    entity_set = set(df.iloc[:, 2].values)
    mp = {e: i for i, e in enumerate(entity_set)}
    indices = [mp[v] for v in df.iloc[:, 2].values]
    return indices
    # return torch.tensor(mapping[df.iloc[:, 2].values])

# Load data
cond_data = load_data_from_directory('./graphs/cond')
proc_data = load_data_from_directory('./graphs/proc')
drug_data = load_data_from_directory('./graphs/drug')

# Create a mapping for each node type (cond, proc, drug)
cond_mapping = map_labels_to_integers(np.concatenate([df.iloc[:, 0].values for df in cond_data]))
proc_mapping = map_labels_to_integers(np.concatenate([df.iloc[:, 0].values for df in proc_data]))
drug_mapping = map_labels_to_integers(np.concatenate([df.iloc[:, 0].values for df in drug_data]))

# Convert the mappings to int32
cond_mapping = cond_mapping.astype(np.int32)
proc_mapping = proc_mapping.astype(np.int32)
drug_mapping = drug_mapping.astype(np.int32)

# Create a heterograph with integer node labels
# graph_data = {
#     ('cond', 'related_to', 'cond'): (),
#     ('proc', 'related_to', 'proc'): (),
#     ('drug', 'related_to', 'drug'): (),
# }
graph_data_cond = {('cond', 'related_to', 'cond'): ()}
graph_data_proc = {('proc', 'related_to', 'proc'): ()}
graph_data_drug = {('drug', 'related_to', 'drug'): ()}


# Save knowledge graphs for cond_data, proc_data, and drug_data
for i, df in enumerate(cond_data):
    try:
        heads: torch.Tensor = map_labels_to_integers_df_h(df, cond_mapping)
        tails: torch.Tensor = map_labels_to_integers_df_t(df, cond_mapping)
    except IndexError:
        continue

    graph_data_cond[('cond', 'related_to', 'cond')] = (heads, tails)

    # Save the knowledge graph for cond_data with a unique name
    with open(f'knowledge_graph_cond_{i}.pkl', 'wb') as f:
        g = dgl.heterograph(graph_data_cond)
        pickle.dump(g, f)

for i, df in enumerate(proc_data):
    try:
        heads: torch.Tensor = map_labels_to_integers_df_h(df, proc_mapping)
        tails: torch.Tensor = map_labels_to_integers_df_t(df, proc_mapping)
    except IndexError:
        continue

    graph_data_proc[('proc', 'related_to', 'proc')] = (heads, tails)

    # Save the knowledge graph for proc_data with a unique name
    with open(f'knowledge_graph_proc_{i}.pkl', 'wb') as f:
        g = dgl.heterograph(graph_data_proc)
        pickle.dump(g, f)

for i, df in enumerate(drug_data):
    try:
        heads: torch.Tensor = map_labels_to_integers_df_h(df, drug_mapping)
        tails: torch.Tensor = map_labels_to_integers_df_t(df, drug_mapping)
    except IndexError:
        continue

    graph_data_drug[('drug', 'related_to', 'drug')] = (heads, tails)

    # Save the knowledge graph for drug_data with a unique name
    with open(f'knowledge_graph_drug_{i}.pkl', 'wb') as f:
        g = dgl.heterograph(graph_data_drug)
        pickle.dump(g, f)

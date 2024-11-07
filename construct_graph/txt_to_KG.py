import pickle
import os
import pandas as pd
import numpy as np
import dgl
import torch

# Function to load data from all txt files in a directory into a list of DataFrames
def load_data_from_directory(directory):
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            df = pd.read_csv(os.path.join(directory, filename), delimiter='\t')
            data_frames.append(df)
    return data_frames

# Load data
cond_data = load_data_from_directory('./graphs/cond')
proc_data = load_data_from_directory('./graphs/proc')
drug_data = load_data_from_directory('./graphs/drug')

# Create a heterograph
graph_data = {
    ('cond', 'related_to', 'cond'): [torch.tensor(df['source'].values) for df in cond_data],
    ('proc', 'related_to', 'proc'): [torch.tensor(df['source'].values) for df in proc_data],
    ('drug', 'related_to', 'drug'): [torch.tensor(df['source'].values) for df in drug_data]
}
g = dgl.heterograph(graph_data)

# Save heterogeneous graph using pickle
with open('knowledge_graph.pkl', 'wb') as f:
    pickle.dump(g, f)







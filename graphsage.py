import dgl
import pickle

# Load the existing DGL heterograph from knowledge_graph.pkl
with open('knowledge_graph.pkl', 'rb') as file:
    graph = pickle.load(file)

# Function to perform data augmentation using GraphSAGE-like sampling for heterograph
def data_augmentation(graph, num_samples_per_type):
    # Empty list to store augmented graphs
    augmented_graphs = []

    for ntype in graph.ntypes:
        # Perform neighbor sampling for each node type
        for node_id in graph.nodes(ntype=ntype):
            sampled_graph = dgl.sampling.sample_neighbors(graph, [node_id], num_samples_per_type[ntype])
            augmented_graphs.append(sampled_graph)

    return augmented_graphs

# Number of samples to take for each node type
num_samples_per_type = {
    'node_type1': 69,
    'node_type2': 180,
    # Add more node types as needed
}

# Perform data augmentation
augmented_graphs = data_augmentation(graph, num_samples_per_type)

# Print the number of augmented graphs
print("Number of augmented graphs:", len(augmented_graphs))

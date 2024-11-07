import os
import pickle
import dgl

def merge_graphs(directory, prefix):
    # Create an empty heterogeneous graph
    merged_graph = dgl.heterograph()

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.pkl'):
            with open(os.path.join(directory, filename), 'rb') as file:
                graph = pickle.load(file)
                # Add nodes and edges from the current graph to the merged graph
                merged_graph = merged_graph + graph

    return merged_graph

# Assuming you have created heterograph objects for each pickle file
merged_graph = merge_graphs('knowledge_graph', 'knowledge_graph_cond')

# Save the merged graph
with open('merged_knowledge_graph.pkl', 'wb') as file:
    pickle.dump(merged_graph, file)

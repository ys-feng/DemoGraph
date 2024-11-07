import dgl
import torch

# Load heterogeneous graphs
g1 = dgl.load_graphs('medical_graph.pkl')[0]
g2 = dgl.load_graphs('mimic3.pkl')[0]

# Merge the graphs
merged_g = dgl.merge([g1, g2])

# The merged graph contains nodes and edges from both original graphs
print('Nodes:', merged_g.num_nodes())
print('Edges:', merged_g.num_edges())

# Node and edge features are updated from g2 if they differ from g1
print(merged_g.nodes['drug'].data['hv'])

# Save merged heterogeneous graph
dgl.save_graphs('merged_graph.bin', merged_g)
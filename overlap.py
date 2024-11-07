import os
import pickle
import dgl
import numpy as np

def load_knowledge_graph():
    """
    Loads the combined knowledge graph from all .KPL files in the knowledge_graph directory.

    :return: Set of nodes in the combined knowledge graph
    """
    knowledge_graph_nodes = set()
    knowledge_graph_directory = '/home/r10user16/GraphAug/knowledge_graph'
    for filename in os.listdir(knowledge_graph_directory):
        if filename.endswith(".KPL"):
            with open(os.path.join(knowledge_graph_directory, filename), 'rb') as fin:
                knowledge_graph_data = pickle.load(fin)
                knowledge_graph_nodes.update(knowledge_graph_data['nodes'])
    return knowledge_graph_nodes

def load_mimic3_data():
    """
    Loads mimic3 data from mimic3_dp.pkl file and returns the set of nodes.

    :return: Set of nodes in the mimic3 knowledge graph
    """
    with open('/home/r10user16/GraphAug/data/mimic3_objects/mimic3_dp.pkl', 'rb') as fin:
        mimic3_data = pickle.load(fin)
        mimic3_nodes = set(mimic3_data['nodes'])
    return mimic3_nodes

# Main program entry
if __name__ == "__main__":
    # Load knowledge graph nodes
    knowledge_graph_nodes = load_knowledge_graph()
    mimic3_nodes = load_mimic3_data()

    # Calculate node overlap
    node_overlap = len(knowledge_graph_nodes & mimic3_nodes)

    print("Node overlap between Knowledge Graph and Mimic3:", node_overlap)

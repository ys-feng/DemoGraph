
import os
import pickle
from collections import OrderedDict
import csv


import dgl
from tqdm import tqdm
import wandb


import torch
from torch.nn import functional as F


import numpy as np
import pandas as pd

import plotly.graph_objects as go

from .trainer import Trainer
from parse import (
   parse_optimizer,
   parse_gnn_model,
   parse_loss
)


from data import load_graph
from utils import acc, metrics


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
       super().__init__(config)


       self.initialize_logger(self.config["name"])


       self.config_gnn = config["GNN"]


       # Initialize GNN model and optimizer
       self.tasks = ["readm"]


       # Load graph, labels and splits
       graph_path = self.config_data["graph_path"]
       dataset_path = self.config_data["dataset_path"]
       labels_path = self.config_data["labels_path"]
       entity_mapping = self.config_data["entity_mapping"]
       edge_dict_path = self.config_data["edge_dict_path"]
       self.graph, self.labels, self.train_mask, self.test_mask = load_graph(graph_path, labels_path)


       with open(edge_dict_path, "rb") as f:
           self.edge_dict = pickle.load(f)


       with open(entity_mapping, "rb") as f:
           self.entity_mapping = pickle.load(f)


       # Transform the graph
       self.graph = dgl.AddReverse()(self.graph)


       # Read node_dict
       self.node_dict = {}
       for tp in self.graph.ntypes:
           self.node_dict.update({tp: torch.arange(self.graph.num_nodes(tp))})


       self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
       self.optimizer = parse_optimizer(self.config_optim, self.gnn)


    def train(self) -> None:
       print(f"Start training GNN")


       training_range = tqdm(range(self.n_epoch), nrows=3)


       for epoch in training_range:
           self.gnn.train()
           self.anneal_temperature(epoch)
           epoch_stats = {"Epoch": epoch + 1}
           preds, labels = None, None
           losses = []


           #Comment below for baseline method,
           #also change out = h[g.ndata["_TYPE"] == 5] into 4 in models\GAT.py

           # TODO: Load list of entities to augment [cond1, drug2, ..., proc3 ...]
           entities = self.get_augmentation_entities()


           kg_df = pd.concat([self.load_kg(entity) for entity in entities], ignore_index=True)




           # TODO: Merge KG into the EHR graph (self.graph)
           ehr_entity_names = set()


           for t in self.edge_dict.values():
               for name in t[0]:
                   ehr_entity_names.add(name)
               for name in t[1]:
                   ehr_entity_names.add(name)


           self.graph = self.merge_ehr_kg(kg_df, ehr_entity_names)


           # Update self.node_dict to include indices of all node types in the new graph
           for tp in self.graph.ntypes:
               self.node_dict.update({tp: torch.arange(self.graph.num_nodes(tp))})


           # kg_df = self.load_kg('0521')

           #Comment above for baseline method

           epoch_stats = {"Epoch": epoch + 1}
           preds, labels = None, None


           # Perform aggregation on visits
           self.optimizer.zero_grad()
           d = self.node_dict.copy()
           for t in self.tasks:
               all_preds = []
               indices = self.train_mask[t]
               d["visit"] = self.node_dict["visit"][indices]
               sg = self.graph.subgraph(d).to(self.device)
               preds = self.gnn(sg, "visit", t)
               labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)
               if t == "drug_rec":
                   preds = preds / self.temperature * 10  # Scale predictions by temperature
                   loss = F.binary_cross_entropy_with_logits(preds, labels)
               else:
                   preds /= self.temperature  # Scale predictions by temperature
                   loss = F.cross_entropy(preds, labels)


           loss.backward()
           self.optimizer.step()


           train_metrics = metrics(preds, labels, "readm")


           # Perform validation and testing
           self.checkpoint_manager.save_model(self.gnn.state_dict())
           test_metrics = self.evaluate()


           training_range.set_description_str("Epoch {} | loss: {:.4f}| Train AUC: {:.4f} | Test AUC: {:.4f} | Test ACC: {:.4f} ".format(
               epoch, loss.item(), train_metrics["tr_accuracy"], test_metrics["test_accuracy"], test_metrics["test_roc_auc"]))


           epoch_stats.update({"Train Loss: ": loss.item()})
           epoch_stats.update(train_metrics)
           epoch_stats.update(test_metrics)


           # Log metrics to Wandb
           wandb.log(epoch_stats)


           # State dict of the model including embeddings
           self.checkpoint_manager.write_new_version(
               self.config,
               self.gnn.state_dict(),
               epoch_stats
           )


           # Remove previous checkpoint
           self.checkpoint_manager.remove_old_version()

    def evaluate(self):
        self.gnn.eval()
        for t in self.tasks:
            indices = self.test_mask[t]
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)


            d = self.node_dict.copy()
            d["visit"] = self.node_dict["visit"][indices]
            sg = self.graph.subgraph(d).to(self.device)
            with torch.no_grad():
                preds = self.gnn(sg, "visit", t)

        self.save_graph(sg, t)

        test_metrics = metrics(preds, labels, t, prefix="test")

        return test_metrics


    def get_masks(self, g: dgl.DGLGraph, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
            labels = [self.labels[task][v] for v in masks]
        else:
            masks = self.test_mask[task]
            labels = [self.labels[task][v] for v in masks]

        m = {}

        for tp in g.ntypes:
            if tp == "visit":
                m[tp] = torch.from_numpy(masks.astype("int32"))
            else:
                m[tp] = torch.zeros(0)

        return m


    def get_labels(self, train: bool, task: str):
       if train:
           masks = self.train_mask[task]
           labels = [self.labels[task][v] for v in masks]
       else:
           masks = self.test_mask[task]
           labels = [self.labels[task][v] for v in masks]

       return masks, labels


    def up_sample(self, scores, label):
       """
       Up sample labels to ensure data balance
       :param scores:
       :param label:
       :return:
       """
       pass


    @staticmethod
    def load_kg(entity):
        # Initialize df to None
        df = None

        file_path = '/home/r10user16/GraphAug/txts/datasetlevel_mimiciii_100_triples.txt'
        try:
            with open(file_path, 'r') as file:
                df = pd.read_csv(file, delimiter='\t', header=None)
        except pd.errors.ParserError as e:
            print(f"ParserError in file: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: No columns to parse from file at {file_path}")
        except Exception as e:  # General exception handling
            print(f"An unknown error occurred: {e}")

        return df

    @staticmethod
    def create_entity_mapping(kg_edges, ehr_entity_names):
       all_entities = set()
       for edge in kg_edges:
           all_entities.add(edge['procedure'])
           all_entities.add(edge['concept'])
           all_entities.update(ehr_entity_names)

       return {entity: index for index, entity in enumerate(all_entities)}


    def merge_ehr_kg(self, kg_df, ehr_entity_names):
       # Convert the DataFrame to a list of dictionaries
       kg_edges = [{'procedure': row[0], 'newrelation': row[1], 'concept': row[2]} for row in kg_df.values]


       # Create mapping for all entities including edge_dict values
       entity_mapping = self.create_entity_mapping(kg_edges, ehr_entity_names)


       # Copy the existing edge dictionary from the object
       edge_dict = self.edge_dict.copy()


       # Convert all string values in edge_dict to integers using entity_mapping, but keep keys as strings
       new_edge_dict = {}
       for key, value in edge_dict.items():
           sub_dict_key0 = key[0]  # Key to identify the correct sub dictionary based on the value at index 0
           sub_dict_key2 = key[2]  # Key to identify the correct sub dictionary based on the value at index 2
           sub_dict0 = self.entity_mapping.get(sub_dict_key0, {})
           sub_dict2 = self.entity_mapping.get(sub_dict_key2, {})
           new_edge_dict[key] = (
               [sub_dict0[entity] for entity in value[0]],
               [sub_dict2[entity] for entity in value[1]]
           )


       # Load procedure_dict from the provided CSV file
       procedure_mapping_file = "/home/r10user16/GraphAug/physionet.org/mimiciii/D_ICD_PROCEDURES2.csv"
       procedure_dict = {}
       with open(procedure_mapping_file, newline='') as csvfile:
           reader = csv.DictReader(csvfile)
           for row in reader:
               procedure_dict[row['name'].lower()] = row['code']


       # Prepare the procedure sub-dictionary for the next part
       procedure_sub_dict = self.entity_mapping.get('procedure', {})


       # Use mapping to convert entities in KG edges to integers and group them by newrelation
       # grouped_edges = {}
       new_procedures = []
       new_concepts = []


       for edge in kg_edges:
           # Check which entity corresponds to a procedure in procedure_dict
           # To ensure 'procedure' and 'concept' are in string type
           if not isinstance(edge['procedure'], str) or not isinstance(edge['concept'], str):
               continue


           procedure_candidate1 = edge['procedure'].lower()
           procedure_candidate2 = edge['concept'].lower()


           procedure_code1 = procedure_dict.get(procedure_candidate1)
           procedure_code2 = procedure_dict.get(procedure_candidate2)


           if procedure_code1 is not None:
               procedure = procedure_sub_dict.get(procedure_code1, procedure_code1)
               concept = entity_mapping[edge['concept']]
               new_procedures.append(procedure)
               new_concepts.append(concept)
           elif procedure_code2 is not None:
               procedure = procedure_sub_dict.get(procedure_code2, procedure_code2)
               concept = entity_mapping[edge['procedure']]
               new_procedures.append(procedure)
               new_concepts.append(concept)
           else:
               # Skip this edge if neither entity corresponds to a procedure
               continue


       # Clean the new_procedures and new_concepts lists to ensure all elements are integers
       cleaned_procedures = []
       cleaned_concepts = []


       for procedure, concept in zip(new_procedures, new_concepts):
           if isinstance(procedure, int) and isinstance(concept, int):
               cleaned_procedures.append(procedure)
               cleaned_concepts.append(concept)


       # Update the entry in new_edge_dict
       new_edge_dict[('procedure', 'newrelation', 'concept')] = (cleaned_procedures, cleaned_concepts)




       # Create the merged DGL heterogeneous graph using the updated edge dictionary
       merged_graph = dgl.heterograph(new_edge_dict)



       # Copy node data from the original graph
       for key, value in self.graph.ndata.items():
           merged_graph.ndata[key] = value


       feat_dim = 128
       for tp in merged_graph.ntypes:
           n_nodes = merged_graph.num_nodes(tp)


           # Initialize features
           feat = torch.randn(n_nodes, feat_dim)
           merged_graph.nodes[tp].data["feat"] = feat


       # Return the merged graph
       return merged_graph


    def visualize_embeddings(self):
        layout = go.Layout(
            autosize=False,
            width=600,
            height=600)
        fig = go.Figure(layout=layout)
        embeddings = self.gnn.embeddings.detach().cpu().numpy()

        from sklearn.manifold import Isomap, TSNE

        offset = 0
        for k, v in self.node_dict.items():
            indices = [i for i in range(offset, offset + 250)]
            tsne = TSNE(n_components=2)
            embeddings_2d = tsne.fit_transform(embeddings[indices])
            offset += len(v)

            fig.add_trace(go.Scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], mode='markers', name=k))

        wandb.log({"chart": fig})


    def get_augmentation_entities(self):
       """
       Get a list of EHR entities for KG retrieval
       e.g., [cond1, proc1, drug2, ...]
       """
       # TODO: Get/sample a set of cond+drug+proc names from self.edge_dict
       # TODO： Call this function to the __init__, assign fetchable names in __init__ to self

       # selected_entities = set(self.edge_dict.keys())

       # TODO: graph_gen.ipynb to generate KGs for diganosis (D_ICD_DIAGNOSIS.csv) and procedures (D_ICD_PROCEDURE.csv), PRESCRIPTIONS.csv
       ehr_entity_names = set()


       for t in self.edge_dict.values():
           for name in t[0]:
               ehr_entity_names.add(name)
           for name in t[1]:
               ehr_entity_names.add(name)


       # kg_entities = set(self.edge_dict.keys())


       kg_entity_names = set()

       # Path to the directory
       directory_path = "/home/r10user16/GraphAug/txts/full_D_ICD_PROCEDURES"


       # Read all files with the .txt extension in the specified directory
       kg_entity_names = {os.path.splitext(filename)[0] for filename in os.listdir(directory_path) if
                          filename.endswith('.txt')}


       selected_entities = [entity for entity in kg_entity_names if entity in ehr_entity_names]


       # TODO: Sample entities
       return selected_entities

    def save_graph(self, g, task):
        with open(f'{self.checkpoint_manager.path}/graph_{task}.pkl', 'wb') as outp:
            pickle.dump(g.cpu(), outp, pickle.HIGHEST_PROTOCOL)
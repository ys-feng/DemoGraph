
import os
import pickle
from collections import OrderedDict
import csv
import random
from typing import List, Any

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

from dgl import LaplacianPE
from dgl import RandomWalkPE
from dgl import DropEdge
from dgl import DropNode


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
       super().__init__(config)

       self.task = self.config["task"]
       self.tasks = [self.task]


       self.initialize_logger(self.config["name"])


       self.config_gnn = config["GNN"]
       self.laplacian_pe_config = config.get("laplacian_pe", {})
       self.drop_edge_config = config.get("drop_edge", {})
       self.drop_node_config = config.get("drop_node", {})
       self.rw_pe_config = config.get("random_walk_pe", {})

       # Initialize GNN model and optimizer
       # self.tasks = ["los"]


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

       # self.conv_label_keys_to_names()

       # Transform the graph with 4 augmenters
       self.graph = dgl.AddReverse()(self.graph)


       # Read node_dict
       self.node_dict = {}
       for tp in self.graph.ntypes:
           self.node_dict.update({tp: torch.arange(self.graph.num_nodes(tp))})

       # Apply Augmenters
       # self.apply_graph_transformations(config)

       self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
       self.optimizer = parse_optimizer(self.config_optim, self.gnn)

    def apply_graph_transformations(self, config):
        # DropNode transformation
        if config.get("drop_node", {}).get("use_drop_node", False):
            drop_prob = config["drop_node"].get("drop_probability", 0.5)
            self.graph = DropNode(drop_prob)(self.graph)

        # DropEdge transformation
        if config.get("drop_edge", {}).get("use_drop_edge", False):
            drop_prob = config["drop_edge"].get("drop_probability", 0.5)
            self.graph = DropEdge(drop_prob)(self.graph)

        # RandomWalkPE transformation
        if config.get("random_walk_pe", {}).get("use_random_walk_pe", False):
            rw_steps = config["random_walk_pe"].get("random_walk_steps", 16)
            self.graph = RandomWalkPE(rw_steps)(self.graph)

        # LaplacianPE transformation
        if config.get("laplacian_pe", {}).get("use_laplacian_pe", False):
            k = config["laplacian_pe"].get("laplacian_pe_k", 4)
            self.graph = dgl.to_bidirected(self.graph)
            self.graph = LaplacianPE(k, feat_name='PE')(self.graph)

    def update_node_indices(self, graph):
        self.node_dict = {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes}

    def validate_node_indices(self, graph, node_dict):
        for ntype, indices in node_dict.items():
            max_index = graph.number_of_nodes(ntype)
            if not all(idx < max_index for idx in indices):
                raise ValueError(f"Invalid node index detected for node type {ntype}")

    def train(self) -> None:
       print(f"Start training GNN")

       graph_copy = self.graph.clone()

       self.apply_graph_transformations(self.config)

       self.update_node_indices(graph_copy)

       self.validate_node_indices(graph_copy, self.node_dict)

       training_range = tqdm(range(self.n_epoch), nrows=3)


       for epoch in training_range:
           self.gnn.train()
           self.anneal_temperature(epoch)
           epoch_stats = {"Epoch": epoch + 1}
           preds, labels = None, None
           # losses = []

           # Comment below for baseline method,
           #also change preds = self.gnn(sg, pred_type=5, task=t) into 4
           # # TODO: Load list of entities to augment [cond1, drug2, ..., proc3 ...]
           # entities = self.get_augmentation_entities()
           # kg_df = pd.concat([self.load_kg(entity) for entity in entities], ignore_index=True)

           # llm_concept_path = '/home/r10user16/GraphAug/txts/datasetlevel_mimiciii_100.txt'
           # llm_concepts = []
           # with open(llm_concept_path, 'r') as file:
           #     for line in file:
           #         llm_concepts.append(line.strip())
           # num_concepts = self.config["GNN"]["num_concepts"]
           # llm_concepts = llm_concepts[:num_concepts]
           #
           # concept_path = '/home/r10user16/GraphAug/txts/datasetlevel_mimiciii_100_triples.txt'
           #
           # kg_df = pd.read_csv(concept_path, sep='\t', header=None)
           #
           # #filter the kg_df to adapt to num_concepts
           # filtered_rows = []
           # for index, row in kg_df.iterrows():
           #     if row[0] in llm_concepts and row[2] in llm_concepts:
           #         filtered_rows.append(row)
           #
           # kg_df = pd.DataFrame(filtered_rows, columns=kg_df.columns)


           # # TODO: Merge KG into the EHR graph (self.graph)
           # ehr_entity_names = set()
           #
           #
           # for t in self.edge_dict.values():
           #     ehr_entity_names.update(t[0])
           #     ehr_entity_names.update(t[1])
           #
           #
           # graph_copy = self.merge_ehr_kg(kg_df, ehr_entity_names, graph_copy)
           # graph_copy = dgl.AddReverse()(graph_copy)    # where roc curve stops dying
           #
           # # Update all labels in self.graph into string-type
           #
           # # Update self.node_dict to include indices of all node types in the new graph
           # for tp in graph_copy.ntypes:
           #     self.node_dict.update({tp: torch.arange(graph_copy.num_nodes(tp))})

           #Comment above for baseline method

           # Perform aggregation on visits
           self.optimizer.zero_grad()
           d = self.node_dict.copy()

           # self.drop_node(d)

           for t in self.tasks:
               all_preds = []
               # indices = self.train_mask[t]
               indices, labels = self.get_indices_labels(t)
               d["visit"] = self.node_dict["visit"][indices]
               # sg = self.graph.subgraph(d).to(self.device)
               sg = graph_copy.subgraph(d).to(self.device) # use graph copy

               # labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)


               preds = self.gnn(sg, pred_type=4, task=t)

               self.save_graph(graph_copy, t)

               # self.save_attn(getattr(self.gnn, "attn"), t)

               if t == "drug_rec":
                   preds = preds / self.temperature * 10  # Scale predictions by temperature
                   loss = F.binary_cross_entropy_with_logits(preds, labels)
               else:
                   preds /= self.temperature  # Scale predictions by temperature
                   loss = F.cross_entropy(preds, labels)

           loss.backward()
           self.optimizer.step()


           train_metrics = metrics(preds, labels, t)


           # Perform validation and testing
           self.checkpoint_manager.save_model(self.gnn.state_dict())
           test_metrics = self.evaluate()

           # test_roc_auc = test_metrics.get("test_roc_auc")
           if t in ["readm", "mort_pred"]:
               training_range.set_description_str(
                   "Epoch {} | loss: {:.4f}| Train AUC: {:.4f} | Test AUC: {:.4f} | Test ACC: {:.4f} ".format(
                       epoch, loss.item(), train_metrics["tr_accuracy"], test_metrics["test_roc_auc"],
                       test_metrics["test_accuracy"]))
           elif t == "los":
               training_range.set_description_str(
                   "Epoch {} | loss: {:.4f}| Train AUC OVO: {:.4f} | Test AUC OVO: {:.4f} | Train F1 Weighted: {:.4f} | Test ACC: {:.4f} ".format(
                       epoch, loss.item(), train_metrics["tr_roc_auc_weighted_ovo"],
                       test_metrics["test_roc_auc_weighted_ovo"], train_metrics["tr_f1_weighted"],
                       test_metrics["test_accuracy"]))
           elif t == "drug_rec":
               training_range.set_description_str(
                   "Epoch {} | loss: {:.4f}| Train ROC AUC Samples: {:.4f} | Test ROC AUC Samples: {:.4f} | Train PR AUC Samples: {:.4f} | Train F1 Weighted: {:.4f} | Train Jaccard Weighted: {:.4f} | Test ACC: {:.4f} ".format(
                       epoch, loss.item(), train_metrics["tr_roc_auc_samples"], test_metrics["test_roc_auc_samples"],
                       train_metrics["tr_pr_auc_samples"], train_metrics["tr_f1_weighted"],
                       train_metrics["tr_jaccard_weighted"],
                       test_metrics["test_accuracy"]))

           epoch_stats.update({"Train Loss: ": loss.item()})
           epoch_stats.update(train_metrics)
           epoch_stats.update(test_metrics)


           # Log metrics to Wandb
           wandb.log(epoch_stats)

           # for i in range(self.gnn.num_layers):
           #     attention_scores = self.graph.edata[f'a_{i}']
           #     attention_scores_list = attention_scores.tolist()
           #     wandb.log({f'Attention Scores Layer {i}': attention_scores_list})


           # State dict of the model including embeddings
           self.checkpoint_manager.write_new_version(
               self.config,
               self.gnn.state_dict(),
               epoch_stats
           )


           # Remove previous checkpoint
           self.checkpoint_manager.remove_old_version()

    def drop_node(self, node_dict_to_modify):
        for node_type, tensor in node_dict_to_modify.items():
            # 将tensor转换为numpy数组
            indices = tensor.numpy()
            # 计算要删除的元素数量
            num_to_drop = int(0.1 * len(indices))

            # 如果没有足够的元素可以删除，则跳过
            if num_to_drop == 0:
                continue

            # 随机选择要删除的索引
            indices_to_drop = np.random.choice(range(len(indices)), size=num_to_drop, replace=False)

            # 创建一个新的索引列表，排除掉被删除的索引
            new_indices = np.delete(indices, indices_to_drop)

            # 将新的索引列表转换回tensor
            node_dict_to_modify[node_type] = torch.tensor(new_indices, dtype=tensor.dtype)


    def evaluate(self):
        self.gnn.eval()
        for t in self.tasks:
            indices, labels = self.get_indices_labels(t, train=False)

            d = self.node_dict.copy()
            # d.pop("concepts")
            d["visit"] = self.node_dict["visit"][indices]
            # sg = self.graph.subgraph(d).to(self.device)
            sg = self.graph.subgraph(d).to(self.device) # use graph copy to evaluate?
            with torch.no_grad():
                preds = self.gnn(sg, pred_type=4, task=t)

        # self.visualize_embeddings()

        # self.save_graph(sg, t)

        test_metrics = metrics(preds, labels, t, prefix="test")

        return test_metrics

    def merge_ehr_kg(self, kg_df, ehr_entity_names, graph):
        #re-establish entity mapping:
        entity_mapping = self.entity_mapping.copy()
        llm_concept_path = '/home/r10user16/GraphAug/txts/datasetlevel_mimiciii_100.txt'
        entity_mapping = self.add_llm_concepts_to_mapping(entity_mapping, llm_concept_path)

        # Convert the DataFrame to a list of dictionaries
        kg_edges = [{'procedure': row[0], 'newrelation': row[1], 'concept': row[2]} for row in kg_df.values]

        # Create mapping for all entities including edge_dict values
        # entity_mapping = self.create_entity_mapping(kg_edges, ehr_entity_names)

        # Copy the existing node and edge dictionary from the object
        edge_dict = self.edge_dict.copy()

        # Convert all string values in edge_dict to integers using entity_mapping, but keep keys as strings
        new_edge_dict = {}
        for key, value in edge_dict.items():
            sub_dict_key0 = key[0]  # Key to identify the correct sub dictionary based on the value at index 0
            sub_dict_key2 = key[2]  # Key to identify the correct sub dictionary based on the value at index 2
            sub_dict0 = self.entity_mapping.get(sub_dict_key0, {}) #removed self.
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
        new_concept_head = []
        new_concept_tail = []

        for edge in kg_edges:
            # Check which entity corresponds to a procedure in procedure_dict
            # To ensure 'procedure' and 'concept' are in string type
            if not isinstance(edge['procedure'], str) or not isinstance(edge['concept'], str):
                continue

            concept_head = entity_mapping['concept'].get(edge['procedure'], edge['procedure'])
            concept_tail = entity_mapping['concept'].get(edge['concept'], edge['concept'])
            new_concept_head.append(concept_head)
            new_concept_tail.append(concept_tail)

        # Clean the new_procedures and new_concepts lists to ensure all elements are integers
        cleaned_concept_head = []
        cleaned_concept_tail = []

        for concept_head, concept_tail in zip(new_concept_head, new_concept_tail):
            if isinstance(concept_head, int) and isinstance(concept_tail, int):
                cleaned_concept_head.append(concept_head)
                cleaned_concept_tail.append(concept_tail)

        llm_concept_path = '/home/r10user16/GraphAug/txts/datasetlevel_mimiciii_100.txt'
        llm_concepts = []
        mapped_concepts = []
        with open(llm_concept_path, 'r') as file:
            for line in file:
                llm_concepts.append(line.strip())
        num_concepts = self.config["GNN"]["num_concepts"]
        llm_concepts = llm_concepts[:num_concepts]
        for concept in llm_concepts:
            mapped_concept = entity_mapping['concept'].get(concept, concept)
            mapped_concepts.append(mapped_concept)

        entity_node_dict = list(self.node_dict.items())
        # sample 'num_concepts' concepts from each subdict of self.node_dict
        for name, tensor in entity_node_dict:
            values = tensor.tolist()
            if len(values) > num_concepts:
                sampled_values = random.sample(values, num_concepts)
            else:
                sampled_values = values
            new_edge_dict[(name, 'newrelation', 'concepts')] = (sampled_values, mapped_concepts)

        # Update the entry in new_edge_dict
        new_edge_dict[('concepts', 'newrelation', 'concepts')] = (cleaned_concept_head, cleaned_concept_tail)

        # Create the merged DGL heterogeneous graph using the updated edge dictionary
        merged_graph = dgl.heterograph(new_edge_dict)

        # Copy node data from the original graph
        for key, value in self.graph.ndata.items():
            merged_graph.ndata[key] = value

        # Deal with feat
        for tp in merged_graph.ntypes:
            # Check if the 'feat' attribute exists for the node type tp
            if 'feat' in merged_graph.nodes[tp].data:
                continue
            else:
                n_nodes = merged_graph.num_nodes(tp)
                feat = torch.randn(n_nodes, 128)
                merged_graph.nodes[tp].data['feat'] = feat

        # Save new entity mapping
        self.save_entity_mapping(entity_mapping)

        # feat_dim = 128
        # for tp in merged_graph.ntypes:
        #     n_nodes = merged_graph.num_nodes(tp)
        #
        #     # Initialize features
        #     feat = torch.randn(n_nodes, feat_dim)
        #     merged_graph.nodes[tp].data["feat"] = feat

        # Return the merged graph
        return merged_graph

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

    @staticmethod
    def add_llm_concepts_to_mapping(entity_mapping, llm_concept_path):
        llm_concepts = []
        with open(llm_concept_path, 'r') as file:
            [llm_concepts.append(line.strip()) for line in file]

        concept_sub_dict = {concept: index for index, concept in enumerate(llm_concepts)}
        entity_mapping['concept'] = concept_sub_dict

        return entity_mapping

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
            indices = [i for i in range(offset, offset + 99)]
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

    def conv_label_keys_to_names(self):

        rev_entity_mapping = {v: k for k, v in self.entity_mapping["visit"].items()}

        for t in self.tasks:
            indices = np.concatenate((self.train_mask[t], self.test_mask[t]))
            names = [rev_entity_mapping[i] for i in indices]

            self.labels[t] = {n: l for n, l in
                              zip(names, self.labels[t].values())}

    def index_to_names(self, indices, e_type="visit"):

        names = [
            key for key, value in self.entity_mapping[e_type].items()
            if value in indices
        ]

        return names
    def save_graph(self, g, task):
        with open(f'{self.checkpoint_manager.path}/graph_{task}.pkl', 'wb') as outp:
            pickle.dump(g.cpu(), outp, pickle.HIGHEST_PROTOCOL)


    def save_attn(self, attn, task):
        with open(f'{self.checkpoint_manager.path}/attention_score_{task}.pkl', 'wb') as outp:
            pickle.dump(attn.detach().cpu(), outp, pickle.HIGHEST_PROTOCOL)

    def save_entity_mapping(self, em):
        with open(f'{self.checkpoint_manager.path}/entity_mapping.pkl', 'wb') as outp:
            pickle.dump(em, outp, pickle.HIGHEST_PROTOCOL)

    def get_indices_labels(self, t, train=True):
        indices = self.train_mask[t] if train else self.test_mask[t]

        if t == "drug_rec":
            all_drugs = self.train_mask["all_drugs"]
            labels = []
            for i in indices:
                drugs = self.labels[t][i]
                labels.append([1 if d in drugs else 0 for d in all_drugs])
            labels = torch.FloatTensor(labels).to(self.device)

        else:
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

        if t == "mort_pred" and train:
            indices = self.down_sample(indices, labels)
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

        return indices, labels

    def down_sample(self, indices, labels):
        """
        Down sample labels to ensure data balance
        :param scores:
        :param label:
        :return:
        """
        n = len(labels[labels == 0])
        neg_indices = indices[labels.detach().cpu() == 0]
        pos_indices = indices[labels.detach().cpu() == 1]
        indices = np.random.choice(len(neg_indices), size=len(pos_indices), replace=True)
        neg_indices = neg_indices[indices]

        return np.concatenate(
            [pos_indices, neg_indices]
        )
from collections import OrderedDict
from tqdm import tqdm
import dgl
import os
import torch
import torch.nn.functional as F
import pandas as pd
from .trainer import Trainer
from parse import parse_optimizer, parse_gnn_model
from utils import acc, metrics
import random
import wandb
from dgl import LaplacianPE
from dgl import RandomWalkPE
from dgl import DropEdge
from dgl import DropNode

class GNNPPITrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.new_nodes = None
        self.n_new_nodes = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initialize_logger(self.config["name"])
        self.config_gnn = config["GNN"]
        self.config_optim = config["optimizer"]
        self.tasks = ["classification"]
        self.laplacian_pe_config = config.get("laplacian_pe", {})
        self.drop_edge_config = config.get("drop_edge", {})
        self.drop_node_config = config.get("drop_node", {})
        self.rw_pe_config = config.get("random_walk_pe", {})
        self.cga_config = config.get("cga", {})

        # Load PPI dataset
        self.train_dataset = dgl.data.PPIDataset(mode='train')
        self.val_dataset = dgl.data.PPIDataset(mode='valid')
        self.test_dataset = dgl.data.PPIDataset(mode='test')

        # Basic initializations

        self.transformed_train_dataset = []
        for idx, graph in enumerate(self.train_dataset):
            graph = graph.to(self.device)


            # Initialize mapping table for node names to IDs for each graph
            self.node_name_to_id = {str(i): i for i in range(graph.number_of_nodes())}

            # Basic initializations
            graph.ndata['train_mask'] = torch.ones(graph.number_of_nodes(), dtype=torch.bool).to(self.device)
            graph.ndata['label'] = graph.ndata['label'].to(self.device)

            # DropNode transformation
            if self.drop_node_config.get("use_node_edge", False):
                drop_prob = self.drop_node_config.get("drop_probability", 0.5)
                drop_node_transform = DropNode(p=drop_prob)
                graph = drop_node_transform(graph)

            # DropEdge transformation
            if self.drop_edge_config.get("use_drop_edge", False):
                drop_prob = self.drop_edge_config.get("drop_probability", 0.5)
                drop_edge_transform = DropEdge(p=drop_prob)
                graph = drop_edge_transform(graph)

            # RandomWalkPE transformation
            if self.rw_pe_config.get("use_random_walk_pe", False):
                rw_steps = self.rw_pe_config.get("random_walk_steps", 16)
                rw_pe_transform = RandomWalkPE(k=rw_steps)
                graph = rw_pe_transform(graph)

            # LaplacianPE transformation
            if self.laplacian_pe_config.get("use_laplacian_pe", False):
                original_features = graph.ndata['feat'] if 'feat' in graph.ndata else None
                graph = graph.to('cpu')
                k = self.laplacian_pe_config.get("laplacian_pe_k")
                graph = dgl.to_bidirected(graph)
                lap_pe = LaplacianPE(k, feat_name='PE')
                graph = lap_pe(graph)
                graph = graph.to(self.device)
                if original_features is not None:
                    graph.ndata['feat'] = original_features


            # Apply CGAug
            if self.cga_config.get("use_cga", False):
                self.augment_graph(graph,
                                   "/home/r10user16/GraphAug/txts/datasetlevel_ppi_25_triples.txt",
                                   "/home/r10user16/GraphAug/txts/datasetlevel_ppi_25_triples.txt")

            graph = dgl.add_self_loop(graph)

            self.transformed_train_dataset.append(graph)

        # Initialize GNN model
        self.gnn = parse_gnn_model(self.config_gnn, self.train_dataset[0], self.tasks).to(self.device)

        self.optimizer = parse_optimizer(self.config_optim, self.gnn)



    def augment_graph(self, graph, node_filepath, edge_filepath):
        new_nodes = self.read_new_nodes(node_filepath)
        self.add_new_nodes_to_graph(graph, new_nodes)
        # self.connect_new_nodes(graph, new_nodes)
        self.add_new_edges(graph, edge_filepath)
        self.n_new_nodes = len(new_nodes)
        self.new_nodes = new_nodes

    def read_new_nodes(self, filepath):
        new_nodes = set()
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().strip('[]').split(', ')
                if len(parts) == 3:
                    head, relation, tail = parts
                    new_nodes.add(head)
                    new_nodes.add(tail)
        new_nodes = list(new_nodes)
        print("Number of new nodes read from file:", len(new_nodes))
        return new_nodes

    def add_new_nodes_to_graph(self, graph, new_nodes):
        # node_name_to_id = graph.node_name_to_id
        num_existing_nodes = graph.number_of_nodes()
        for i, node_name in enumerate(new_nodes):
            self.node_name_to_id[node_name] = num_existing_nodes + i
        # self.graph.add_nodes(num_new_nodes)
        # Add new features if necessary
        # self.graph.ndata['feat'] = torch.cat([self.graph.ndata['feat'], new_features], dim=0)
        assert graph.ndata['feat'].shape[0] == graph.num_nodes(), "Mismatch in node and feature count"

    def connect_new_nodes(self, graph_copy):    # NEW Update: introduce number of edges per node
        num_existing_nodes = graph_copy.number_of_nodes() - self.n_new_nodes
        nec = self.config_gnn.get("nec", 100)

        for new_node_id in range(num_existing_nodes, graph_copy.number_of_nodes()):
            target_node_ids = random.sample(range(num_existing_nodes), min(nec, num_existing_nodes))

            for target_node_id in target_node_ids:
                graph_copy.add_edges(new_node_id, target_node_id)
                graph_copy.add_edges(target_node_id, new_node_id)
        return graph_copy

    def add_new_edges(self, graph, filepath):
        source_nodes, target_nodes = self.read_new_edges(graph, filepath)
        graph.add_edges(source_nodes, target_nodes)
        print(f"Added {len(source_nodes)} new edges to the graph")

    def read_new_edges(self, graph, filepath):
        source_nodes = []
        target_nodes = []
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().strip('[]').split(', ')
                if len(parts) == 3:
                    head, _, tail = parts
                    source_nodes.append(self.convert_to_node_id(head))
                    target_nodes.append(self.convert_to_node_id(tail))
        return source_nodes, target_nodes

    def convert_to_node_id(self, node_name):
        return self.node_name_to_id.get(node_name)

    def train(self):
        print(f"Start training GNN")
        training_range = tqdm(range(self.n_epoch), ncols=100)

        for epoch in training_range:
            self.gnn.train()
            self.anneal_temperature(epoch)

            training_range = tqdm(range(self.n_epoch), ncols=100)
            total_loss = 0
            total_train_acc = 0

            for graph in self.transformed_train_dataset:
                self.optimizer.zero_grad()
                graph = graph.to(self.device)
                graph_copy = graph.clone()
                graph_copy = self.connect_new_nodes(graph_copy)
                features = graph_copy.ndata['feat']
                labels = graph_copy.ndata['label']

                preds = self.gnn(graph_copy, features, "classification")
                preds = preds / self.temperature

                loss = F.binary_cross_entropy_with_logits(preds[graph_copy.ndata['train_mask']],
                                                          labels[graph_copy.ndata['train_mask']])
                # loss = F.binary_cross_entropy(
                #     preds[torch.cat([self.train_mask,
                #                      torch.zeros(self.n_new_nodes, device=self.device).type_as(self.train_mask)])
                #     ],
                #     labels
                # )
                # loss = F.binary_cross_entropy_with_logits(preds[train_mask], labels[train_mask])

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()


                # Metrics
                # train_acc = calculate_accuracy(preds[torch.cat([self.train_mask,
                #                      torch.zeros(self.n_new_nodes, device=self.device).type_as(self.train_mask)])
                #     ], labels)

                # train_acc = calculate_accuracy(preds[train_mask], labels)[train_mask]

                train_acc = calculate_accuracy(preds[graph_copy.ndata['train_mask']], labels[graph_copy.ndata['train_mask']])
                total_train_acc += train_acc

            avg_loss = total_loss / len(self.transformed_train_dataset)
            avg_train_acc = total_train_acc / len(self.transformed_train_dataset)

            # Evaluate the model
            test_acc, pr_auc_samples, test_f1 = self.evaluate()
            # weighted_f1, auroc, accuracy = self.weightedf1()

            # Logging
            print(f"Epoch: {epoch} | Loss: {avg_loss:.4f} | Train ACC: {avg_train_acc:.4f}")
            print(f"Test ACC: {test_acc:.4f} | Test Micro F1: {test_f1:.4f} | Test AUPR: {pr_auc_samples:.4f}")

            wandb.log({"Epoch": epoch + 1, "Train Loss": avg_loss, "Train ACC": avg_train_acc,
                       "Test ACC": test_acc, "Test F1": test_f1,
                       "Test AUPR": pr_auc_samples})


    def evaluate(self):
        self.gnn.eval()
        total_preds, total_true = [], []
        with torch.no_grad():
            for graph in self.test_dataset:
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                preds = self.gnn(graph, features, "classification")
                # preds_binary = torch.sigmoid(preds) > 0.5
                total_preds.append(preds.sigmoid())
                total_true.append(labels)

        all_preds = torch.cat(total_preds, dim=0)
        all_true = torch.cat(total_true, dim=0)

        # micro_f1 = self.calculate_micro_f1(all_preds, all_true)
        met = metrics(all_preds, all_true, "drug_rec", "te", ["accuracy",
                                                              # "roc_auc_samples",
                                                              "pr_auc_samples", "f1_micro"]) # drug_rec for multi-label classification
        accuracy = met["te_accuracy"]
        # roc_auc_samples = met["te_roc_auc_samples"]
        pr_auc_samples = met["te_pr_auc_samples"]
        test_f1 = met["te_f1_micro"]

        return accuracy, pr_auc_samples, test_f1
              # roc_auc_samples unable to calculate due to ValueError



    @staticmethod
    def calculate_micro_f1(preds, labels):
        preds_int = preds.int()  # Converting Boolean values to integers
        labels_int = labels.int()

        true_positives = (preds_int & labels_int).sum(axis=0).float()
        predicted_positives = preds_int.sum(axis=0).float()
        actual_positives = labels_int.sum(axis=0).float()

        precision = true_positives.sum() / (predicted_positives.sum() + 1e-10)
        recall = true_positives.sum() / (actual_positives.sum() + 1e-10)
        micro_f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return micro_f1.item()

def calculate_accuracy(logits, labels):
    # Using a sigmoid function since multi-label classification can be treated as multiple independent binary classifications
    predictions = torch.sigmoid(logits) > 0.5
    correct = (predictions == labels).float()
    # Calculate accuracy by considering how many labels per instance were correctly predicted
    instance_accuracy = correct.mean(dim=1)
    overall_accuracy = instance_accuracy.mean()
    return overall_accuracy.item()

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


class GNNCiteseerTrainer(Trainer):
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


        # Load Citeseer graph, labels, and splits
        dataset = dgl.data.CiteseerGraphDataset()
        self.graph = dataset[0].to(self.device)
        self.labels = self.graph.ndata['label'].to(self.device)
        self.train_mask = self.graph.ndata['train_mask'].to(self.device)
        self.val_mask = self.graph.ndata['val_mask'].to(self.device)
        self.test_mask = self.graph.ndata['test_mask'].to(self.device)
        self.graph_eva = self.graph.clone()

        # Initialize mapping table for node names to IDs
        self.node_name_to_id = {str(i): i for i in range(self.graph.number_of_nodes())}

        # DropNode transformation
        if self.drop_node_config.get("use_node_edge", False):
            drop_prob = self.drop_node_config.get("drop_probability", 0.5)
            drop_node_transform = DropNode(p=drop_prob)
            graph = drop_node_transform(self.graph)

        # DropEdge transformation
        if self.drop_edge_config.get("use_drop_edge", False):
            drop_prob = self.drop_edge_config.get("drop_probability", 0.5)
            drop_edge_transform = DropEdge(p=drop_prob)
            graph = drop_edge_transform(self.graph)

        # RandomWalkPE transformation
        if self.rw_pe_config.get("use_random_walk_pe", False):
            rw_steps = self.rw_pe_config.get("random_walk_steps", 16)
            rw_pe_transform = RandomWalkPE(k=rw_steps)
            graph = rw_pe_transform(self.graph)

        # LaplacianPE transformation
        if self.laplacian_pe_config.get("use_laplacian_pe", False):
            original_features = self.graph.ndata['feat'] if 'feat' in self.graph.ndata else None
            self.graph = self.graph.to('cpu')
            k = self.laplacian_pe_config.get("laplacian_pe_k")
            self.graph = dgl.to_bidirected(self.graph)
            lap_pe = LaplacianPE(k, feat_name='PE')
            self.graph = lap_pe(self.graph)
            self.graph = self.graph.to(self.device)
            if original_features is not None:
                self.graph.ndata['feat'] = original_features


        # Apply CGAug
        if self.cga_config.get("use_cga", False):
            self.augment_graph("/home/r10user16/GraphAug/txts/datasetlevel_ppi_50_triples.txt",
                               "/home/r10user16/GraphAug/txts/datasetlevel_ppi_50_triples.txt")

        self.graph = dgl.add_self_loop(self.graph)

        # Initialize GNN model
        self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

    def augment_graph(self, node_filepath, edge_filepath):
        new_nodes = self.read_new_nodes(node_filepath)
        self.n_new_nodes = len(new_nodes)
        self.new_nodes = new_nodes
        self.add_new_nodes_to_graph(new_nodes)
        # self.connect_new_nodes(new_nodes)
        self.add_new_edges(edge_filepath)


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

    def add_new_nodes_to_graph(self, new_nodes):    #NEW updated.
        num_existing_nodes = self.graph.number_of_nodes()
        # num_new_nodes = len(new_nodes)
        for i, node_name in enumerate(new_nodes):
            self.node_name_to_id[node_name] = num_existing_nodes + i
        # self.graph.add_nodes(num_new_nodes)
        # Add new features if necessary
        # self.graph.ndata['feat'] = torch.cat([self.graph.ndata['feat'], new_features], dim=0)
        assert self.graph.ndata['feat'].shape[0] == self.graph.num_nodes(), "Mismatch in node and feature count"

    def connect_new_nodes(self, graph_copy):    # NEW Update: introduce number of edges per node
        #Dynamic merging
        num_existing_nodes = graph_copy.number_of_nodes() - self.n_new_nodes
        nec = self.config_gnn.get("nec", 100)

        for new_node_id in range(num_existing_nodes, graph_copy.number_of_nodes()):
            target_node_ids = random.sample(range(num_existing_nodes), min(nec, num_existing_nodes))

            for target_node_id in target_node_ids:
                graph_copy.add_edges(new_node_id, target_node_id)
                graph_copy.add_edges(target_node_id, new_node_id)
        return graph_copy

    def add_new_edges(self, filepath):
        source_nodes, target_nodes = self.read_new_edges(filepath)
        self.graph.add_edges(source_nodes, target_nodes)
        print(f"Added {len(source_nodes)} new edges to the graph")

    def read_new_edges(self, filepath):
        source_nodes = []
        target_nodes = []
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().strip('[]').split(', ')
                if len(parts) == 3:
                    head, relation, tail = parts
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
            self.optimizer.zero_grad()
            epoch_stats = {"Epoch": epoch + 1}

            graph_copy = self.graph.clone()

            if self.cga_config.get("use_cga", False):
                graph_copy = self.connect_new_nodes(graph_copy)

            preds = self.gnn(graph_copy, graph_copy.ndata['feat'], "classification")
            preds = preds / self.temperature
            labels = self.labels[self.train_mask].to(self.device)

            # loss = F.cross_entropy(
            #     preds[torch.cat([self.train_mask,
            #                      torch.zeros(self.n_new_nodes, device=self.device).type_as(self.train_mask)])
            #     ],
            #     labels
            # )
            loss = F.cross_entropy(preds[self.train_mask], labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            # train_acc = calculate_accuracy(preds[torch.cat([self.train_mask,
            #                      torch.zeros(self.n_new_nodes, device=self.device).type_as(self.train_mask)])
            #     ], labels)
            train_acc = calculate_accuracy(preds[self.train_mask], labels)

            test_acc = self.evaluate()

            test_f1, auroc = self.weightedf1()



            # Logging
            training_range.set_description_str(
                f"Epoch {epoch} | loss: {loss.item():.4f} | Train ACC: {train_acc:.4f} | "
                f"Test ACC: {test_acc:.4f} | Test Weighted F1: {test_f1:.4f} | Test AUC: {auroc:.4f}")

            wandb.log({"Epoch": epoch + 1, "Train Loss": loss.item(), "Train ACC": train_acc,
                       "Test ACC": test_acc, "Test F1": test_f1, "Test AUROC": auroc})

    def evaluate(self):
        self.gnn.eval()
        with torch.no_grad():
            preds = self.gnn(self.graph_eva, self.graph_eva.ndata['feat'], task="classification")
        labels = self.labels[self.test_mask].to(self.device)

        preds = preds[self.test_mask]

        test_acc = calculate_accuracy(preds, labels)

        # test_acc = calculate_accuracy(preds[torch.cat([self.test_mask,
        #                          torch.zeros(self.n_new_nodes, device=self.device).type_as(self.test_mask)])
        #         ], labels)
        # test_acc = calculate_accuracy(preds[self.test_mask], labels)
        return test_acc

    def weightedf1(self):
        self.gnn.eval()
        with torch.no_grad():
            preds = self.gnn(self.graph_eva, self.graph_eva.ndata['feat'], "classification")

            num_new_nodes = self.graph_eva.number_of_nodes() - self.test_mask.shape[0]
            extended_test_mask = torch.cat(
                [self.test_mask, torch.zeros(num_new_nodes, dtype=torch.bool, device=self.device)])

            labels = self.labels[self.test_mask].to(self.device)
            preds_test = preds[extended_test_mask]

            preds_test_softmax = F.softmax(preds_test, dim=1)

            # calculate weighted F1 and auroc. unable to calculate auprc.
            met = metrics(preds_test_softmax, labels, "los", "te", ["f1_weighted", "roc_auc_weighted_ovo"])
            weighted_f1 = met["te_f1_weighted"]
            auroc = met["te_roc_auc_weighted_ovo"]

            return weighted_f1, auroc


def calculate_accuracy(logits, labels):
    _, predicted_classes = torch.max(logits, dim=1)
    correct = (predicted_classes == labels).float()
    accuracy = correct.mean()
    return accuracy.item()
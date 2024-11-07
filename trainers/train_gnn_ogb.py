from collections import OrderedDict
from tqdm import tqdm
import dgl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from .trainer import Trainer
from parse import parse_optimizer, parse_gnn_model
from utils import acc, metrics
import random
import wandb
from dgl.dataloading import NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import GraphDataLoader
from torch import nn
from dgl.nn import SAGEConv, GATConv, GraphConv
from dgl import LaplacianPE
from dgl import RandomWalkPE
from dgl import DropEdge
from dgl import DropNode

import numpy as np
import sklearn
import os

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h


class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        in_feats = config['in_dim']
        hidden_dim = config['hidden_dim']
        num_classes = config['out_dim']
        heads = config['num_heads']
        attn_drop = config.get('attn_drop', 0.6)
        feat_drop = config.get('feat_drop', 0.6)
        negative_slope = config.get('negative_slope', 0.2)

        for i in range(len(heads)):
            out_feats = hidden_dim if i < len(heads) - 1 else num_classes
            num_heads = heads[i]
            in_feats_layer = in_feats if i == 0 else hidden_dim * heads[i - 1]

            layer = GATConv(
                in_feats=in_feats_layer,
                out_feats=out_feats,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                activation=F.elu if i < len(heads) - 1 else None,
                allow_zero_in_degree=True
            )
            self.layers.append(layer)

    def forward(self, mfgs, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            # Ensure the correct graph is passed for each layer
            graph = mfgs[i]  # Assuming mfgs is a list of graphs, one for each layer
            h = layer(graph, h)
            if i < len(self.layers) - 1:
                # In all but the last layer, use concatenation
                h = h.flatten(1)
            else:
                # In the last layer, use mean (if not concatenating)
                h = h.mean(dim=1)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, h_feats, activation=F.relu))
        for i in range(1, num_layers - 1):
            self.layers.append(GraphConv(h_feats, h_feats, activation=F.relu))
        self.layers.append(GraphConv(h_feats, num_classes))
        self.dropout = dropout

    def forward(self, mfgs, features):
        h = features
        for i, (layer, mfg) in enumerate(zip(self.layers, mfgs)):
            if i != 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(mfg, h)
        return h


class GNNOGBTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.new_nodes = None
        self.n_new_nodes = 0
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



        # Load OGB dataset
        dataset = DglNodePropPredDataset(name='ogbn-products')
        self.graph = dataset.graph[0].to(self.device)
        self.graph = dgl.add_self_loop(self.graph)
        self.labels = dataset[0][1].to(self.device)
        num_classes = (self.labels.max() + 1).item()
        self.graph.ndata['labels'] = self.labels

        split_idx = dataset.get_idx_split()
        self.train_mask = split_idx['train'].to(self.device)
        self.val_mask = split_idx['valid'].to(self.device)
        self.test_mask = split_idx['test'].to(self.device)

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

        # Data augmentation
        if self.cga_config.get("use_cga", False):
            self.augment_graph("/home/r10user16/GraphAug/txts/datasetlevel_ogb_45_triples.txt",
                               "/home/r10user16/GraphAug/txts/datasetlevel_ogb_45_triples.txt", "dataset")

        # Initialize GNN model
        # SAGE
        # self.gnn = Model(self.config_gnn['in_dim'], self.config_gnn['hidden_dim'], num_classes).to(self.device)

        #GAT
        # self.gnn = GAT(self.config_gnn).to(self.device)

        #GCN
        self.gnn = GCN(
            self.config_gnn['in_dim'],
            self.config_gnn['hidden_dim'],
            num_classes,
            num_layers=self.config_gnn.get('num_layers', 2),
            dropout=self.config_gnn.get('dropout', 0.5)
        ).to(self.device)

        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

        # Initialize the sampler
        self.sampler = NeighborSampler(
            [4, 4],
            # num_hops=2,
            # batch_size=100,
            # neighbor_type='in',
            # shuffle=True,
            # num_workers=4,
            # add_self_loop=True
        )
        self.dataloader = dgl.dataloading.DataLoader(
            self.graph, self.train_mask, self.sampler,
            batch_size=1024, shuffle=True, drop_last=False, num_workers=0
        )

        self.valid_dataloader = dgl.dataloading.DataLoader(
            self.graph, self.val_mask, self.sampler,
            batch_size=1024, shuffle=True, drop_last=False, num_workers=0
        )



    def augment_graph(self, node_filepath, edge_filepath, granularity):
        new_nodes = self.read_new_nodes(node_filepath)
        self.add_new_nodes_to_graph(new_nodes)
        # self.connect_new_nodes(new_nodes)
        self.add_new_edges(edge_filepath)
        self.n_new_nodes += len(new_nodes)
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

    def add_new_nodes_to_graph(self, new_nodes):    #NEW updated.
        num_existing_nodes = self.graph.number_of_nodes()
        # num_new_nodes = len(new_nodes)
        for i, node_name in enumerate(new_nodes):
            self.node_name_to_id[node_name] = num_existing_nodes + i
        # self.graph.add_nodes(num_new_nodes)
        # Add new features if necessary
        # self.graph.ndata['feat'] = torch.cat([self.graph.ndata['feat'], new_features], dim=0)
        assert self.graph.ndata['feat'].shape[0] == self.graph.num_nodes(), "Mismatch in node and feature count"

    def connect_new_nodes(self, graph_copy, granularity="dataset"):    # NEW Update: introduce number of edges per node
        #Dynamic merging
        num_existing_nodes = graph_copy.number_of_nodes() - len(self.new_nodes)
        nec = self.config_gnn.get("nec", 100)

        edges = []

        for new_node_id in range(num_existing_nodes, graph_copy.number_of_nodes()):

            if granularity == "dataset":
                target_node_ids = random.sample(range(num_existing_nodes), min(nec, num_existing_nodes))
            elif granularity == "node-type":
                target_node_ids = torch.arange(graph_copy.number_of_nodes(), device=self.device)[graph_copy.ndata["label"] == 1]
            else:
                raise  NotImplementedError("Unsupported Granularity Level")

            for target_node_id in target_node_ids:
                edges.append((new_node_id, target_node_id))
                edges.append((target_node_id, new_node_id))

        graph_copy.add_edges(
            torch.LongTensor([e[0] for e in edges]).to(self.device),
            torch.LongTensor([e[1] for e in edges]).to(self.device)
        )

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

        for epoch in range(1000):
            self.gnn.train()

            with tqdm(self.dataloader) as tq:
                for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                    # feature copy from CPU to GPU takes place here
                    inputs = mfgs[0].srcdata['feat']
                    labels = mfgs[-1].dstdata['labels'].squeeze()

                    predictions = self.gnn(mfgs, inputs)

                    loss = F.cross_entropy(predictions, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(),
                                                              predictions.argmax(1).detach().cpu().numpy())
                    wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

                    tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

                accuracy = self.evaluate()
                    # print(f"Epoch {epoch + 1} | Test Micro F1: {test_f1:.4f}")

    def evaluate(self):

        predictions = []
        labels = []
        with tqdm(self.valid_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata['feat']
                labels.append(mfgs[-1].dstdata['labels'].squeeze().cpu().numpy())
                predictions.append(self.gnn(mfgs, inputs).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)

            wandb.log({"validation_accuracy": accuracy})

        return accuracy

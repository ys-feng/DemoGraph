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

class GNNRedditTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initialize_logger(self.config["name"])
        self.config_gnn = config["GNN"]
        self.config_optim = config["optimizer"]
        self.tasks = ["classification"]

        # Load Reddit dataset
        dataset = dgl.data.RedditDataset()
        self.graph = dataset[0].to(self.device)
        self.labels = self.graph.ndata['label'].to(self.device)
        self.train_mask = self.graph.ndata['train_mask'].to(self.device)
        self.val_mask = self.graph.ndata['val_mask'].to(self.device)
        self.test_mask = self.graph.ndata['test_mask'].to(self.device)

        # Initialize GNN model
        self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

        # Initialize the sampler
        self.sampler = NeighborSampler(self.graph,
                                       num_hops=2,
                                       batch_size=100,
                                       neighbor_type='in',
                                       shuffle=True,
                                       num_workers=4,
                                       add_self_loop=True)

    def train(self):
        for epoch in tqdm(range(self.n_epoch), ncols=100):
            self.gnn.train()
            total_loss = 0

            for nf in self.sampler:  # Loop over the subgraphs
                self.optimizer.zero_grad()
                node_batch = nf.layer_parent_nid(-1).to(self.device)  # Nodes in the current batch
                input_features = nf.blocks[0].srcdata['feat'].to(
                    self.device)  # Input features for the nodes in the batch

                # Forward pass with subgraph and input features
                preds = self.gnn(nf, input_features, "classification")
                loss = F.cross_entropy(preds, self.labels[node_batch][self.train_mask[node_batch]])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.sampler)
            print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")

            # Evaluate the model
            test_f1 = self.evaluate()
            print(f"Epoch {epoch + 1} | Test Micro F1: {test_f1:.4f}")

    def evaluate(self):
        self.gnn.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for nf in self.sampler:  # Loop over the subgraphs
                node_batch = nf.layer_parent_nid(-1).to(self.device)  # Nodes in the current batch
                input_features = nf.blocks[0].srcdata['feat'].to(
                    self.device)  # Input features for the nodes in the batch

                preds = self.gnn(nf, input_features, "classification")
                preds_binary = preds.argmax(dim=1)

                # Collect predictions and labels for nodes in the current batch
                all_preds.append(preds_binary[self.test_mask[node_batch]])
                all_labels.append(self.labels[node_batch][self.test_mask[node_batch]])

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute Micro F1 Score
        micro_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='micro')

        return micro_f1

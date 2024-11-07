"""
GraphSAGE
"""

import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from dgl.nn.pytorch.conv import SAGEConv

from .GNN import GNN

class GraphSAGE(GNN):
    def __init__(self,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 activation,
                 feat_drop,
                 tasks,
                 causal):

        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, feat_drop, tasks, causal)

    def forward(self, g: dgl.DGLHeteroGraph, pred_type, task):

        if task != "classification":
            g = dgl.to_homogeneous(g, ndata=["feat"], store_type=True)
            g = dgl.add_self_loop(g)

        h = g.ndata["feat"]
        h = self.get_logit(g, h)

        # h = self.out[task](h)
        h = self.out[task](g, h)

        if task != "classification":
            out = h[g.ndata["_TYPE"] == pred_type]
        else:
            out = h

        # self.set_embeddings(h)

        return out

    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            # h, a = layer(g, h, get_attention=True) #calculate attention score
            h = layer(g, h)
            if i != len(layers) - 1:
                h = h.flatten(1)
            else:
                h = h.flatten(1)

            # g.edata[f'a_{i}'] = a #store attention score for each layer

        self.set_embeddings(h)

        # self.embeddings = h.detach()

        return h


    # For GraphSAGE model:
    def get_layers(self):

        layers = nn.ModuleList()
        for l in range(self.n_layers):
            if l == 0:
                # input projection (no residual)
                layers.append(SAGEConv(
                    self.in_dim, self.hidden_dim, aggregator_type='mean',
                    feat_drop=self.dor, activation=self.activation))
            else:
                # due to multi-head, the in_dim = num_hidden * num_heads
                layers.append(SAGEConv(
                    self.hidden_dim, self.hidden_dim, aggregator_type='mean',
                    feat_drop=self.dor, activation=self.activation))

        return layers

    def get_layer(self, in_dim, out_dim, l=-1, linear=False, act=True):
        act = None if not act else self.activation

        if not linear:
            return SAGEConv(
                    in_dim, out_dim, aggregator_type='mean',
                    feat_drop=self.dor, activation=act)
        else:
            return nn.Linear(in_dim, out_dim)
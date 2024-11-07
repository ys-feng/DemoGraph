"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn

import dgl
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from dgl.nn.pytorch.conv import SAGEConv

from .GNN import GNN
from dgl import LaplacianPE
from dgl import RandomWalkPE
from dgl import DropEdge
from dgl import DropNode

class GAT(GNN):
    def __init__(self,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 tasks,
                 causal,
                 linear):

        self.heads = heads
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual

        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, feat_drop, tasks, causal, linear)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # <----

    def forward(self, g: dgl.DGLHeteroGraph, pred_type, task):

        if task != "classification":
            g = dgl.to_homogeneous(g, ndata=["feat"], store_type=True)
            g = dgl.add_self_loop(g)

            g = g.to(self.device) # <----

            g = RandomWalkPE(4)(g)
            # g = LaplacianPE(3, feat_name='PE')(g)

        h = g.ndata["feat"]
        h = self.get_logit(g, h)

        h = self.out[task](h) if self.linear else self.out[task](g, h).mean(1)

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
                h = h.mean(1) if self.linear else h.flatten(1)
                # h = h.flatten(1)

            # g.edata[f'a_{i}'] = a #store attention score for each layer

        # self.set_embeddings(h)

        # self.embeddings = h.detach()

        return h


    # For GAT model:
    def get_layers(self):

        layers = nn.ModuleList()
        for l in range(self.n_layers):
            if l == 0:
                # input projection (no residual)
                layers.append(GATConv(
                    self.in_dim, self.hidden_dim, self.heads[0],
                    self.dor, self.attn_drop, self.negative_slope, False, self.activation))
            else:
                # due to multi-head, the in_dim = num_hidden * num_heads
                layers.append(GATConv(
                    self.hidden_dim * self.heads[l-1], self.hidden_dim, self.heads[l],
                    self.dor, self.attn_drop, self.negative_slope, self.residual, self.activation))

        return layers

    def get_layer(self, in_dim, out_dim, l=-1, act=True):
        act = None if not act else self.activation

        if not self.linear:
            return GATConv(
                    in_dim * self.heads[l-1], out_dim, self.heads[l],
                    self.dor, self.attn_drop, self.negative_slope, self.residual, act)
        else:
            return nn.Linear(in_dim, out_dim)

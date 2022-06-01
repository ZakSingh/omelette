from typing import Union, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch_geometric as pyg
import numpy as np
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GATNetwork(nn.Module):
    """A Graph Attentional Network (GAT) which serves as a Policy Network for our RL agent."""

    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 3, hidden_size: int = 128,
                 out_std=np.sqrt(2)):
        super(GATNetwork, self).__init__()
        self.gnn = pyg.nn.GAT(in_channels=num_node_features, hidden_channels=hidden_size,
                              out_channels=hidden_size, num_layers=n_layers, add_self_loops=False,
                              norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                              act="leaky_relu")
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                                  layer_init(nn.Linear(hidden_size, n_actions), std=out_std))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = self.head(x)
        return x


class GINNetwork(nn.Module):

    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 3, hidden_size: int = 128,
                 out_std=np.sqrt(2)):
        super(GINNetwork, self).__init__()
        self.gnn = pyg.nn.GIN(in_channels=num_node_features, hidden_channels=hidden_size,
                              out_channels=hidden_size, num_layers=n_layers,
                              norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                              act="leaky_relu")
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                                  layer_init(nn.Linear(hidden_size, n_actions), std=out_std))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = self.head(x)
        return x


class GCNNetwork(nn.Module):

    def __init__(self, num_node_features: int, n_actions: int, hidden_size: int = 128, out_std=np.sqrt(2)):
        super(GCNNetwork, self).__init__()
        self.gnn = pyg.nn.GCN(in_channels=num_node_features, hidden_channels=hidden_size,
                              out_channels=hidden_size, num_layers=2)
        self.head = layer_init(nn.Linear(hidden_size, n_actions), std=out_std)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = self.head(x)
        return x


class SAGENetwork(nn.Module):
    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 3, hidden_size: int = 128,
                 out_std: float = np.sqrt(2)):
        super(SAGENetwork, self).__init__()
        self.gnn = pyg.nn.GraphSAGE(in_channels=num_node_features, hidden_channels=hidden_size,
                                    out_channels=hidden_size, num_layers=n_layers, act="leaky_relu")

        self.mem1 = pyg.nn.MemPooling(
            hidden_size, hidden_size, heads=4, num_clusters=10)
        self.mem2 = pyg.nn.MemPooling(
            hidden_size, n_actions, heads=4, num_clusters=1)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = F.leaky_relu(x)
        x, S1 = self.mem1(x, data.batch)
        x = F.leaky_relu(x)
        x, S2 = self.mem2(x)
        x = x.squeeze(1)
        return x


class GraphTransformerNetwork(nn.Module):
    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 2, hidden_size: int = 128):
        super(GraphTransformerNetwork, self).__init__()
        self.gnn = pyg.nn.GraphMultisetTransformer(in_channels=num_node_features, hidden_channels=hidden_size,
                                                   out_channels=n_actions,
                                                   num_heads=4,
                                                   layer_norm=True)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index, batch=data.batch)
        return x

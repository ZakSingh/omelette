from torch import nn
from torch_geometric.nn import GAT, global_mean_pool


class GATPolicyNetwork(nn.module):
    """A Graph Attentional Network (GAT) which serves as a Policy Network for our RL agent."""

    def __init__(self, obs_size: int, n_actions: int, dp_rate_linear=0.5, hidden_size: int = 128):
        super(GATPolicyNetwork, self).__init__()
        self.gnn = GAT(in_channels=obs_size, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=2)
        self.head = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(hidden_size, n_actions))

    def forward(self, x, edge_index, edge_attr):
        x = self.net(x, edge_index, edge_attr)
        x = global_mean_pool(x)
        x = self.head(x)
        return x

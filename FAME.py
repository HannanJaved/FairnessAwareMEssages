import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class FairnessAwareMEssages(MessagePassing):
    def __init__(self, in_channels, out_channels, sens_attribute_tensor):
        super(FairnessAwareMEssages, self).__init__(aggr='mean')  
        self.lin = nn.Linear(in_channels, out_channels)
        self.sensitive_attr = sens_attribute_tensor
        self.bias_correction = nn.Parameter(torch.rand(1))

    def forward(self, x, edge_index):        
        # Add self-loops 
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        group_difference = self.sensitive_attr[row] - self.sensitive_attr[col]
        
        # Adjust messages based on statistical parity
        fairness_adjustment = (1 + self.bias_correction * group_difference.view(-1, 1))

        return fairness_adjustment * norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
    
class FAME_GCN(torch.nn.Module):
    def __init__(self, data, sens_attribute_tensor, layers=1, hidden=128, dropout=0):
        super(FAME_GCN, self).__init__()
        self.conv1 = FairnessAwareMEssages(data.num_node_features, hidden, sens_attribute_tensor)
        self.convs = torch.nn.ModuleList()
        
        for i in range(layers - 1):
            self.convs.append(FairnessAwareMEssages(hidden, hidden, sens_attribute_tensor=))
        
        # self.conv2 = FairnessAwareMEssages(hidden, 2)
        self.fc = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x = self.conv2(x, edge_index)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
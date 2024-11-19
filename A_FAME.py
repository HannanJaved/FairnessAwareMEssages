import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class A_FAME(MessagePassing):
    def __init__(self, in_channels, out_channels, sens_attribute_tensor):
        super(A_FAME, self).__init__(aggr='add') 
        self.lin = Linear(in_channels, out_channels) 
        self.att = Linear(2 * out_channels, 1) 
        
        self.sensitive_attr = sens_attribute_tensor 
        self.bias_correction = Parameter(torch.rand(1))  

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j, size_i):
        x_cat = torch.cat([x_i, x_j], dim=-1)  
        alpha = self.att(x_cat)

        row, col = edge_index
        group_difference = self.sensitive_attr[row] - self.sensitive_attr[col]

        fairness_adjustment = self.bias_correction * group_difference.view(-1, 1)
        alpha = alpha + fairness_adjustment

        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)

        return alpha * x_j

    def update(self, aggr_out):
        return aggr_out
    
# GCN class that takes in the data as an input for dimensions of the convolutions
class A_FAME_GAT(torch.nn.Module):
    def __init__(self, data, sens_attribute_tensor, layers=1, hidden=128, dropout=0):
        super(A_FAME_GAT, self).__init__()
        self.conv1 = A_FAME(data.num_node_features, hidden, sens_attribute_tensor)
        self.convs = torch.nn.ModuleList()
        
        for i in range(layers - 1):
            self.convs.append(A_FAME(hidden, hidden, sens_attribute_tensor))
        
        # self.conv2 = A_FAME(hidden, 2)
        self.fc = Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=self.dropout, training=self.training)

        # x = self.conv2(x, edge_index)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
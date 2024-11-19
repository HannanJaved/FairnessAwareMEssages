import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
	def __init__(self, nfeat, nhid=128, nclass=2, dropout=0):
		super(GAT, self).__init__()
		self.body = GAT_Body(nfeat,nhid,dropout)
		self.fc = nn.Linear(nhid, nclass)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, x, edge_index):
		x = self.body(x, edge_index)
		x = self.fc(x)
		return F.log_softmax(x, dim=1)
		

class GAT_Body(nn.Module):
	def __init__(self, nfeat, nhid, dropout):
		super(GAT_Body, self).__init__()
		self.gc1 = GATConv(nfeat, nhid)

	def forward(self, x, edge_index):
		x = self.gc1(x, edge_index)
		return x
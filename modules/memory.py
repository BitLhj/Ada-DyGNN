import torch
import numpy as np
from torch import nn


class Memory(nn.Module):
	def __init__(self, n_nodes, emb_dim, device='cpu'):
		super(Memory, self).__init__()
		self.n_nodes = n_nodes
		self.emb_dim = emb_dim
		self.device = device
		self.__init_memory__()

	def __init_memory__(self, seed=0):
		torch.manual_seed(seed)
		self.emb = nn.Parameter(torch.zeros((self.n_nodes, self.emb_dim)).to(self.device),
								requires_grad=False)
		nn.init.xavier_normal_(self.emb)

	def detach_memory(self):
		self.emb.detach_()

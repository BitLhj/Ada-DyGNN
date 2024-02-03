import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from modules.memory import Memory
from torch.distributions import Bernoulli
from torch.autograd import Variable


class AdaDyGNN(nn.Module):
	def __init__(self, n_nodes, n_neighbors, n_update_neighbors,
				 edge_dim, emb_dim, message_dim,
				 neighbor_finder, edge_feat, tot_time, device='cpu'):
		super(AdaDyGNN, self).__init__()
		torch.manual_seed(0)
		# graph
		self.damp_list = []
		self.n_nodes = n_nodes
		# Dimensions
		self.feat_dim = 100
		self.edge_dim = edge_dim
		self.emb_dim = emb_dim
		self.message_dim = message_dim
		# memory
		self.edge_feat = torch.tensor(edge_feat, dtype=torch.float).to(device)
		self.memory = Memory(self.n_nodes, self.emb_dim, device)
		# other
		self.tot_time = tot_time
		self.n_neighbors = n_neighbors
		self.n_update_neighbors = n_update_neighbors
		self.neighbor_finder = neighbor_finder
		self.device = device
		self.temperature = 5
		self.eps = 1e-10
		self.gamma = 1
		# GAT
		self.W_g = nn.Parameter(torch.zeros((self.emb_dim, self.message_dim // 2)).to(self.device))
		self.a = nn.Parameter(torch.zeros(self.message_dim).to(self.device))
		self.dropout = nn.Dropout(p=0.5)
		nn.init.xavier_normal_(self.W_g)

		self.W_e = nn.Parameter(torch.zeros((self.edge_dim, self.message_dim)).to(self.device))
		nn.init.xavier_normal_(self.W_e)

		self.W_uc = nn.Parameter(torch.zeros((self.emb_dim + 2 * self.message_dim, self.emb_dim)).to(self.device))
		self.W_un = nn.Parameter(torch.zeros((self.emb_dim + self.message_dim, self.emb_dim)).to(self.device))
		nn.init.xavier_normal_(self.W_uc)
		nn.init.xavier_normal_(self.W_un)

		self.W_p = nn.Parameter(torch.zeros((2 * self.message_dim, self.message_dim)).to(self.device))
		nn.init.xavier_normal_(self.W_p)

		self.W_1 = nn.Parameter(torch.zeros((self.emb_dim * 2, self.emb_dim)).to(self.device))
		self.W_2 = nn.Parameter(torch.zeros((self.emb_dim, 1)).to(self.device))
		nn.init.xavier_normal_(self.W_1)
		nn.init.xavier_normal_(self.W_2)
		self.beta = 0.1

	def forward(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
		message = []
		for i, idxs in enumerate([src_idxs, dst_idxs]):
			neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps, self.n_neighbors)
			current_time = np.repeat(np.expand_dims(timestamps, axis=1), self.n_neighbors, axis=1)
			interval = (current_time - edge_times) / self.tot_time
			damp = 1 / (1 + self.gamma * interval)
			neighbors = torch.from_numpy(neighbors).long().to(self.device)
			bs = neighbors.shape[0]
			neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_neighbors, self.emb_dim)
			damp = np.repeat(np.expand_dims(damp, axis=2), self.emb_dim, axis=2)
			neighbor_emb = neighbor_emb * torch.from_numpy(np.float32(damp)).to(self.device)
			# from neighbors
			h_n = torch.matmul(neighbor_emb, self.W_g)
			# from centre node
			h_c = torch.matmul(self.memory.emb[idxs], self.W_g).unsqueeze(dim=1).repeat(1, self.n_neighbors, 1)
			h_in = torch.cat((h_c, h_n), dim=2)
			h_in = self.dropout(h_in)
			att = F.leaky_relu(torch.matmul(h_in, self.a), negative_slope=0.2)
			att = att.softmax(dim=1).unsqueeze(dim=2).repeat(1, 1, self.message_dim // 2)
			h = h_n * att
			message.append(h)
		h = torch.cat((message[0], message[1]), dim=2).sum(dim=1).tanh()
		h_e = torch.matmul(self.edge_feat[edge_idxs], self.W_e).tanh()
		h = torch.cat((h, h_e), dim=1)

		to_updated_src = torch.matmul(torch.cat((self.memory.emb[src_idxs], h), dim=1), self.W_uc).tanh()
		to_updated_dst = torch.matmul(torch.cat((self.memory.emb[dst_idxs], h), dim=1), self.W_uc).tanh()
		self.memory.emb[src_idxs] = to_updated_src
		self.memory.emb[dst_idxs] = to_updated_dst

		# prop
		for idxs in [src_idxs, dst_idxs]:
			neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps,
																				  self.n_update_neighbors)
			current_time = np.repeat(np.expand_dims(timestamps, axis=1), self.n_update_neighbors, axis=1)
			interval = (current_time - edge_times) / self.tot_time
			damp = 1 / (1 + self.gamma * interval)
			neighbors = torch.from_numpy(neighbors).long().to(self.device)
			bs = neighbors.shape[0]
			neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_update_neighbors, self.emb_dim)
			# neighbor_emb [bs][n_update_neighbors][emb_dim]
			damp = np.repeat(np.expand_dims(damp, axis=2), self.emb_dim, axis=2)
			neighbor_emb = neighbor_emb * torch.from_numpy(np.float32(damp)).to(self.device)
			h1 = torch.matmul(h, self.W_p)
			h1 = h1.unsqueeze(dim=1).repeat(1, self.n_update_neighbors, 1)
			h2 = h1 * neighbor_emb

			h2 = h2 / (h2.norm(dim=2).view(bs, self.n_update_neighbors, -1) + self.eps)
			att = torch.softmax(h2.sum(dim=2), dim=1)
			changed_emb = h1 * att.unsqueeze(dim=2).repeat(1, 1, self.emb_dim)

			x = torch.cat((neighbor_emb, changed_emb), dim=2)
			x.detach_()
			x = torch.matmul(x, self.W_1)
			x = x.relu()
			x = torch.matmul(x, self.W_2)
			probs = x.sigmoid()
			changed_emb = torch.matmul(torch.cat((self.memory.emb[neighbors.flatten()],
												  changed_emb.flatten().view(-1, self.emb_dim)), dim=1), self.W_un).tanh()
			policy_loss = 0
			reward = 0
			policy_map = probs.data.clone()
			policy_map[policy_map < 0.5] = 0.0
			policy_map[policy_map >= 0.5] = 1.0
			policy_map = Variable(policy_map)
			distr = Bernoulli(probs)
			policy = distr.sample()
			if not self.training:
				mask = policy_map.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
				self.memory.emb[neighbors.flatten()] = mask * changed_emb + (1 - mask) * self.memory.emb[
					neighbors.flatten()]
			else:
				policy_map = policy_map.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
				policy_sample = policy.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
				ori_emb = self.memory.emb[neighbors.flatten()]
				updated_emb_map = policy_map * changed_emb + (1 - policy_map) * ori_emb
				updated_emb_sample = policy_sample * changed_emb + (1 - policy_sample) * ori_emb
				reward_map = self.get_reward(idxs, updated_emb_map).detach()
				reward_sample = self.get_reward(idxs, updated_emb_sample).detach()
				advantage = reward_sample - reward_map
				loss = -distr.log_prob(policy) * Variable(advantage).expand_as(policy)
				loss = loss.sum()
				probs = probs.clamp(1e-15, 1 - 1e-15)
				entropy_loss = -probs * torch.log(probs)
				entropy_loss = self.beta * entropy_loss.sum()
				loss = (loss - entropy_loss) / bs / self.n_update_neighbors
				self.memory.emb[neighbors.flatten()] = updated_emb_sample
				policy_loss += loss
				reward += reward_sample

		# compute loss
		pos_score, neg_score = self.compute_score(src_idxs, dst_idxs, neg_idxs)
		return pos_score, neg_score, policy_loss, reward

	def get_reward(self, idxs, neighbor_emb):
		central_emb = self.memory.emb[idxs].repeat(1, self.n_update_neighbors).view(-1, self.emb_dim)
		central_emb_norm = F.normalize(central_emb, p=2, dim=1).detach()
		neighbor_emb_norm = F.normalize(neighbor_emb, p=2, dim=1)
		cos_sim = torch.matmul(central_emb_norm, neighbor_emb_norm.t())
		return cos_sim.mean()

	def compute_sim(self, src_idxs):
		src_norm = F.normalize(self.memory.emb[src_idxs], p=2, dim=1)
		emb_norm = F.normalize(self.memory.emb, p=2, dim=1)
		cos_sim = torch.matmul(src_norm, emb_norm.t())
		sorted_cos_sim, idx = cos_sim.sort(descending=True)
		return sorted_cos_sim, idx

	def compute_score(self, src_idxs, dst_idxs, neg_idxs):
		pos_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[dst_idxs], dim=1)
		neg_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[neg_idxs], dim=1)
		return pos_score.sigmoid(), neg_score.sigmoid()

	def reset_graph(self):
		self.memory.__init_memory__()

	def set_neighbor_finder(self, neighbor_finder):
		self.neighbor_finder = neighbor_finder

	def detach_memory(self):
		self.memory.detach_memory()

	def back_up_memory(self):
		return self.memory.emb.clone()

	def restore_memory(self, back_up):
		self.memory.emb = nn.Parameter(back_up)

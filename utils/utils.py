import numpy as np
import torch

class EarlyStopMonitor(object):
	def __init__(self, max_round=20, higher_better=True, tolerance=1e-10):
		self.max_round = max_round
		self.num_round = 0

		self.epoch_count = 0
		self.best_epoch = 0

		self.last_best = None
		self.higher_better = higher_better
		self.tolerance = tolerance

	def early_stop_check(self, curr_val):
		if not self.higher_better:
			curr_val *= -1
		if self.last_best is None:
			self.last_best = curr_val
		elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
			self.last_best = curr_val
			self.num_round = 0
			self.best_epoch = self.epoch_count
		else:
			self.num_round += 1

		self.epoch_count += 1

		return self.num_round >= self.max_round


class RandEdgeSampler(object):
	def __init__(self, src_list, dst_list, seed=None):
		self.seed = None
		self.src_list = np.unique(src_list)
		self.dst_list = np.unique(dst_list)
		self.len_src = len(self.src_list)
		self.len_dst = len(self.dst_list)
		if seed is not None:
			self.seed = seed
			self.random_state = np.random.RandomState(self.seed)

	def sample(self, size):
		if self.seed is None:
			src_idx = np.random.randint(0, self.len_src, size)
			dst_idx = np.random.randint(0, self.len_dst, size)
		else:
			src_idx = self.random_state.randint(0, self.len_src, size)
			dst_idx = self.random_state.randint(0, self.len_dst, size)
		return self.src_list[src_idx], self.dst_list[dst_idx]

	def reset_random_state(self):
		self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform):
	max_node_idx = max(data.src.max(), data.dst.max())
	adj_list = [[] for _ in range(max_node_idx + 1)]
	for src, dst, edge_idx, timestamp in zip(data.src, data.dst, data.edge_idxs,
											 data.timestamps):
		adj_list[src].append((dst, edge_idx, timestamp))
		adj_list[dst].append((src, edge_idx, timestamp))
	return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
	def __init__(self, adj_list, uniform=False, seed=None):
		self.node_to_neighbors = []
		self.node_to_edge_idxs = []
		self.node_to_edge_time_stamps = []
		for neighbors in adj_list:
			# format of neighbors : (neighbor, edge_idx, timestamp)
			# sorted base on timestamp
			sorted_neighbor = sorted(neighbors, key=lambda x: x[2])
			self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbor]))
			self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbor]))
			self.node_to_edge_time_stamps.append(np.array([x[2] for x in sorted_neighbor]))

		self.uniform = uniform
		if seed is not None:
			self.seed = seed
			self.random_state = np.random.RandomState(self.seed)

	def find_before(self, src_idx, cut_time):
		"""
			Extracts all the interactions happening before cut_time
			for user src_idx in the overall interaction graph.
			The returned interactions are sorted by time.

			Returns 3 lists: neighbors, edge_idxs, timestamps

		  """
		i = np.searchsorted(self.node_to_edge_time_stamps[src_idx], cut_time)
		return self.node_to_neighbors[src_idx][:i], \
			   self.node_to_edge_idxs[src_idx][:i], \
			   self.node_to_edge_time_stamps[src_idx][:i]

	def get_temporal_neighbor(self, src_nodes, timestamps, n_neighbors=20):
		"""
			Given a list of users ids and relative cut times, extracts
			a sampled temporal neighborhood of each user in the list.

			Params
			------
			src_idx_l: List[int]
			cut_time_l: List[float],
			num_neighbors: int
		"""
		assert (len(src_nodes)) == len(timestamps)

		tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
		# All interactions described in these matrices are sorted in each row by time
		neighbors = np.zeros((len(src_nodes), tmp_n_neighbors)).astype(np.float32)
		# each entry in position (i,j) represent the id of the item
		# targeted by user src_idx_l[i] with an interaction happening
		# before cut_time_l[i]
		edge_times = np.zeros((len(src_nodes), tmp_n_neighbors)).astype(np.float32)
		# each entry in position (i,j) represent the timestamp of
		# an interaction between user src_idx_l[i] and item neighbors[i,j]
		# happening before cut_time_l[i]
		edge_idxs = np.zeros((len(src_nodes), tmp_n_neighbors)).astype(np.int32)
		# each entry in position (i,j) represent the interaction index
		# of an interaction between user src_idx_l[i] and item neighbors[i,j]
		# happening before cut_time_l[i]

		for i, (src_node, timestamp) in enumerate(zip(src_nodes, timestamps)):
			src_neighbors, src_edge_idxs, src_edge_times = \
				self.find_before(src_node, timestamp)
			# extracts all neighbors, interactions indexes and timestamps
			# of all interactions of user source_node happening before cut_time
			if len(src_neighbors) > 0 and n_neighbors > 0:
				if self.uniform:
					# if we are applying uniform sampling, shuffles the data
					# above before sampling
					sample_idx = np.random.randint(0, len(src_neighbors), n_neighbors)

					neighbors[i, :] = src_neighbors[sample_idx]
					edge_times[i, :] = src_edge_times[sample_idx]
					edge_idxs[i, :] = src_edge_idxs[sample_idx]
					# re-sort based on time
					pos = edge_times[i, :].argsort()
					neighbors[i, :] = neighbors[i, :][pos]
					edge_times[i, :] = edge_times[i, :][pos]
					edge_idxs[i, :] = edge_idxs[i, :][pos]
				else:
					# Take most recent interactions
					source_edge_times = src_edge_times[-n_neighbors:]
					source_neighbors = src_neighbors[-n_neighbors:]
					source_edge_idxs = src_edge_idxs[-n_neighbors:]
					assert (len(source_neighbors) <= n_neighbors)
					assert (len(source_edge_times) <= n_neighbors)
					assert (len(source_edge_idxs) <= n_neighbors)

					neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
					edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
					edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
		return neighbors, edge_idxs, edge_times

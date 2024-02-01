import torch
import numpy as np
import pandas as pd
import random


class Data:
	def __init__(self, src, dst, timestamps, edge_idxs, labels):
		self.src = src
		self.dst = dst
		self.timestamps = timestamps
		self.edge_idxs = edge_idxs
		self.labels = labels
		self.n_interactions = len(src)
		self.unique_nodes = set(src) | set(dst)
		self.n_unique_nodes = len(self.unique_nodes)


def get_data(DatasetName):
	graph = pd.read_csv("./data/ml_{}.csv".format(DatasetName))
	edge_features = np.load("./data/ml_{}.npy".format(DatasetName))
	node_features = np.load("./data/ml_{}_node.npy".format(DatasetName))

	val_time, test_time = list(np.quantile(graph.ts, [0.8, 0.9]))

	src = graph.u.values
	dst = graph.i.values
	edge_idxs = graph.idx.values
	labels = graph.label.values
	timestamps = graph.ts.values

	full_data = Data(src, dst, timestamps, edge_idxs, labels)

	random.seed(2020)
	node_set = set(src) | set(dst)
	n_total_nodes = len(node_set)

	test_node_set = set(src[timestamps > val_time]) | (set(dst[timestamps > val_time]))

	new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_nodes)))
	new_test_src_mask = graph.u.map(lambda x: x in new_test_node_set).values
	new_test_dst_mask = graph.i.map(lambda x: x in new_test_node_set).values

	observed_edges_mask = np.logical_and(~new_test_src_mask, ~new_test_dst_mask)

	train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
	train_data = Data(src[train_mask], dst[train_mask], timestamps[train_mask],
					  edge_idxs[train_mask], labels[train_mask])
	train_node_set = set(train_data.src) | set(train_data.dst)

	assert len(train_node_set & new_test_node_set) == 0

	new_node_set = node_set - train_node_set

	val_mask = np.logical_and(timestamps > val_time, timestamps <= test_time)
	test_mask = timestamps > test_time

	edge_contains_new_node_mask = np.array(
		[a in new_node_set or b in new_node_set for a, b in zip(src, dst)])
	new_node_val_mask = np.logical_and(edge_contains_new_node_mask, val_mask)
	new_node_test_mask = np.logical_and(edge_contains_new_node_mask, test_mask)

	val_data = Data(src[val_mask], dst[val_mask], timestamps[val_mask],
					edge_idxs[val_mask], labels[val_mask])
	test_data = Data(src[test_mask], dst[test_mask], timestamps[test_mask],
					 edge_idxs[test_mask], labels[test_mask])
	new_node_val_data = Data(src[new_node_val_mask], dst[new_node_val_mask],
							 timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
							 labels[new_node_val_mask])
	new_node_test_data = Data(src[new_node_test_mask], dst[new_node_test_mask],
							  timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
							  labels[new_node_test_mask])

	print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
																				 full_data.n_unique_nodes))
	print("The training dataset has {} interactions, involving {} different nodes".format(
		train_data.n_interactions, train_data.n_unique_nodes))
	print("The validation dataset has {} interactions, involving {} different nodes".format(
		val_data.n_interactions, val_data.n_unique_nodes))
	print("The test dataset has {} interactions, involving {} different nodes".format(
		test_data.n_interactions, test_data.n_unique_nodes))
	print("The new node validation dataset has {} interactions, involving {} different nodes".format(
		new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
	print("The new node test dataset has {} interactions, involving {} different nodes".format(
		new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
	print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
		len(new_test_node_set)))

	return node_features, edge_features, full_data, train_data, \
		   val_data, test_data, new_node_val_data, new_node_test_data,timestamps[-1]


def computer_time_statics(src, dst, timestamps):
	last_timestamp_src = dict()
	last_timestamp_dst = dict()
	all_timediffs_src = []
	all_timediffs_dst = []
	for k in range(len(src)):
		src_id = src[k]
		dst_id = dst[k]
		cur_timestamp = timestamps[k]
		if src_id not in last_timestamp_src.keys():
			last_timestamp_src[src_id] = 0
		if dst_id not in last_timestamp_dst.keys():
			last_timestamp_dst[dst_id] = 0
		all_timediffs_src.append(cur_timestamp - last_timestamp_src[src_id])
		all_timediffs_dst.append(cur_timestamp - last_timestamp_dst[dst_id])
		last_timestamp_src[src_id] = cur_timestamp
		last_timestamp_dst[dst_id] = cur_timestamp
	assert len(all_timediffs_src) == len(src)
	assert len(all_timediffs_dst) == len(dst)
	mean_time_shift_src = np.mean(all_timediffs_src)
	std_time_shift_src = np.std(all_timediffs_src)
	mean_time_shift_dst = np.mean(all_timediffs_dst)
	std_time_shift_dst = np.std(all_timediffs_dst)
	return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

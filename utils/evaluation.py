import numpy as np
import torch
import math
from sklearn.metrics import average_precision_score, roc_auc_score

def eval_edge_prediction(model, negative_edge_sampler, data, bs):
	assert negative_edge_sampler.seed is not None
	negative_edge_sampler.reset_random_state()
	val_mrr, val_recall_20, val_recall_50 = [], [], []
	with torch.no_grad():
		num_batch = math.ceil(len(data.src) / bs)
		for i in range(num_batch):
			if i == 0:
				show_details = False
			else:
				show_details = False
			st_idx = i * bs
			ed_idx = min((i + 1) * bs, len(data.src))
			src_batch = data.src[st_idx:ed_idx]
			dst_batch = data.dst[st_idx:ed_idx]
			edge_batch = data.edge_idxs[st_idx:ed_idx]
			timestamps_batch = data.timestamps[st_idx:ed_idx]
			size = len(src_batch)
			_, negative_batch = negative_edge_sampler.sample(size)

			src_cos_sim, src_idx = model.compute_sim(src_batch)
			dst_cos_sim, dst_idx = model.compute_sim(dst_batch)
			recall_20 = recall(dst_batch, src_idx, 20) + recall(src_batch, dst_idx, 20)
			recall_50 = recall(dst_batch, src_idx, 50) + recall(src_batch, dst_idx, 50)
			mrr = MRR(dst_batch, src_idx) + MRR(src_batch, dst_idx)
			val_recall_20.append(recall_20 / 2)
			val_recall_50.append(recall_50 / 2)
			val_mrr.append(mrr / 2)
			model(src_batch, dst_batch, negative_batch, edge_batch, timestamps_batch)

	return np.mean(val_mrr), np.mean(val_recall_20), np.mean(val_recall_50)


def recall(dst_idxs, idx, top_k):
	bs = idx.shape[0]
	idx = idx[:, :top_k]
	rec = np.array([a in idx[i].cpu() for i, a in enumerate(dst_idxs)])
	rec = rec.sum() / rec.size
	return rec


def MRR(dst_idxs, idx):
	bs = idx.shape[0]
	mrr = np.array([float(np.where(idx[i].cpu() == a)[0] + 1) for i, a in enumerate(dst_idxs)])
	mrr = (1 / mrr).mean()
	return mrr

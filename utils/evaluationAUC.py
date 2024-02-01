import numpy as np
import torch
import math
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction_auc(model, negative_edge_sampler, data):
	assert negative_edge_sampler.seed is not None
	negative_edge_sampler.reset_random_state()
	val_acc, val_ap, val_auc = [], [], []
	bs = 30
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
			pos_prob, neg_prob, _, _ = model(src_batch, dst_batch, negative_batch, edge_batch, timestamps_batch)
			pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
			pred_label = pred_score > 0.5
			true_label = np.concatenate([np.ones(size), np.zeros(size)])

			val_acc.append((pred_label == true_label).mean())
			val_ap.append(average_precision_score(true_label, pred_score))
			# val_f1.append(f1_score(true_label, pred_label))
			val_auc.append(roc_auc_score(true_label, pred_score))
	return np.mean(val_acc), np.mean(val_ap), np.mean(val_auc)

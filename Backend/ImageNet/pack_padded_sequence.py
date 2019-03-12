import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


def pack_padded_sequence(sorted_tracks, lengths, batch_first=False, enforce_sorted=True):
	lengths = torch.as_tensor(lengths, dtype=torch.int64)
	lengths = lengths.cpu()
	if enforce_sorted:
		sorted_indices = None
	else:
		lengths, sorted_indices = torch.sort(lengths, descending=True)
		sorted_indices = sorted_indices.to(sorted_tracks.device)
		batch_dim = 0 if batch_first else 1
		sorted_tracks = sorted_tracks.index_select(batch_dim, sorted_indices)

	data, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(
		sorted_tracks, lengths, batch_first)
	return PackedSequence(data, batch_sizes)
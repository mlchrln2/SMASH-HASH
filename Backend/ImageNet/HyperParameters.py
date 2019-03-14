options = {
	'learning_rate' : 1e-4,
	'batch_size'    : 16,
	'num_epochs'    : 100,
	'channel_size'  : 512,
	'vocab_size'    : 29550,
	'embed_size'    : 512,
	'hidden_size'   : 512,
	'bidirectional' : False,
	'num_layers'    : 1,
	'max_len'	    : 25,
	'align_size'    : 100,
	'window'        : (7,7),
	'start_word'    : '<START>',
	'end_word'      : '<END>',
	'unk_word'      : '<UNK>',
	'data_dir'      : '../../CocoDataset'
}
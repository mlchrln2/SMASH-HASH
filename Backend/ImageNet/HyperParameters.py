options = {
	'learning_rate' : 1e-3,
	'batch_size'    : 4,
	'num_epochs'    : 100,
	'channel_size'  : 3,
	'vocab_size'    : 29549,
	'embed_size'    : 512,
	'hidden_size'   : 256,
	'bidirectional' : True,
	'num_layers'    : 1,
	'max_len'	    : 25,
	'align_size'    : 100,
	'window'        : (51,51),
	'start_word'    : '<START>',
	'end_word'      : '<END>',
	'unk_word'      : '<UNK>',
	'data_dir'      : '../../CocoDataset'
}
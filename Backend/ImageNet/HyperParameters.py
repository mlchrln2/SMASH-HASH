options = {
	'learning_rate' : 1e-4,
	'batch_size'    : 32,
	'num_epochs'    : 120,
	'channel_size'  : 512,
	'vocab_size'    : 29550,
	'embed_size'    : 512,
	'hidden_size'   : 512,
	'max_len'	    : 20,
	'drop'			: 0.5,
	'window'        : (11,11),
	'start_word'    : '<START>',
	'end_word'      : '<END>',
	'unk_word'      : '<UNK>',
	'data_dir'      : '../../CocoDataset'
}
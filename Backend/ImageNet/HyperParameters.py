'''Parameters used during the training and testing stages'''

OPTIONS = {
    'learning_rate': 1e-4,  # the rate at which to update the gradient
    'batch_size': 32,  # the number of batches to use for one optimizer update
    'num_epochs': 120,  # the number of times to train on the dataset before completing
    'channel_size': 512,  # the size of the image channels
    'vocab_size': 13900,  # the size of the vocab (first run DataWriter.py)
    'embed_size': 512,  # the embeding size to use for an index caption
    'hidden_size': 512,  # the output size of the lstm
    'max_len': 20,  # maximum length of an infered caption
    'drop': 0.5,  # drop out rate
    'window': (7, 7),  # the window used for the LocalAttention2d model
    'beam_size': 10,  # the number of paths to take during the beam search
    'start_word': '<START>',  # the token to use for the start_word
    'end_word': '<END>',  # the token to use for the end word
    'unk_word': '<UNK>',  # the token to used for the unknown words
    'data_dir': '../../CocoDataset',  # the location of the coco dataset and caption
    'itr_save': 600  # how many iterations to run before saving the model
}

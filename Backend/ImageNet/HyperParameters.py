'''Parameters used during the training and testing stages'''

OPTIONS = {
    'data_dir': '../../CocoDataset',  # the location of the coco dataset and caption.
    'itr_save': 6,  # the number of iterations to run before saving the model.
    'start_word': '<START>',  # the token to use for the start_word.
    'end_word': '<END>',  # the token to use for the end word.
    'unk_word': '<UNK>',  # the token to used for the unknown words.
    'learning_rate': 1e-4,  # the rate used to update the gradient.
    'batch_size': 128,  # the number of batches to use for one optimizer update.
    'num_epochs': 120,  # the number of times to train on the dataset before completing.
    'channel_size': 3,  # the size of the feature maps.
    'vocab_size': 13900,  # the size of the vocab (first run DataWriter.py).
    'embed_size': 512,  # the embeding size to use for an index caption.
    'drop': 0.5,  # the drop out rate.
    'hidden_size': 512,  # the output size of the rnn layer.
    'window': (51, 51),  # the window used for the 2d local attention model.
    'max_len': 20,  # the maximum length of an inferred caption.
    'beam_size': 10,  # the number of paths to check during the beam search.
    'momentum': 0.01, # the momentum used for the batch normalization
}

'''Main method for training the network outlined in NeuralNetworks.py and Attention.py'''

# dependencies
import gc
import sys
import os
import torch
from tensorboardX import SummaryWriter

# user defined modules
from HyperParameters import OPTIONS
from NeuralNetworks import Image2Caption
from DataLoader import TRAIN_LOADER as TRAIN_DATA
from DataLoader import VAL_LOADER as VAL_DATA
from pack_padded_sequence import pack_padded_sequence

# load in hyper-parameters from python file
MODEL_DIR = OPTIONS['model_dir']
NUM_EPOCHS = OPTIONS['num_epochs']
ITR_SAVE = OPTIONS['itr_save']
DEVICE = OPTIONS['device']

# initialize model and loss function
MODEL = None

#create model directory if it does not currently exist
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

''' to continue training a previously saved model type the model directory as a parameter'''

try:
    CHECKPOINT = torch.load(sys.argv[1])
    print('Training saved model...')
    MODEL = CHECKPOINT['model']
    START_EPOCH = CHECKPOINT['epoch']
except:
    MODEL = Image2Caption()
    print('Training new model...')
    START_EPOCH = -1

print('Note model parameters:\n{}'.format(MODEL.parameters))

# set the mode to train
MODEL.train()

# create a logger
WRITER = SummaryWriter()
'''
WRITER.add_graph(MODEL,
(torch.randn(1,3,224,224), torch.tensor([[0]]),torch.tensor([1])),verbose=True)
'''

'''train model and update the weights using the pack padded outputs with the pack padded labels.
Save the model every ITR_SAVE iterations so that the program can be killed if needed.'''
for epoch in range(START_EPOCH + 1, NUM_EPOCHS):
    for i, (img, labels, lengths) in enumerate(TRAIN_DATA):
        gc.collect()
        MODEL.optimizer.zero_grad()
        predictions = MODEL(img, labels[:, :-1], lengths - 1)
        predictions = pack_padded_sequence(
            predictions, lengths - 1, batch_first=True)[0]
        labels = pack_padded_sequence(
            labels[:, 1:], lengths - 1, batch_first=True)[0]
        loss = MODEL.criterion(predictions, labels)
        loss.backward()
        MODEL.optimizer.step()
        del loss, labels, predictions, img, lengths
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        print('epoch {} of {} --- iteration {} of {}'.format(epoch + 1, NUM_EPOCHS,
                                                             i + 1, len(TRAIN_DATA)),
              end='\r')
    torch.save({'epoch': epoch,
                'model': MODEL
                }, '{}/img_embedding_model_{}.pth'.format(MODEL_DIR, epoch))
    '''
    error = 0
    for i, (img, labels, lengths) in enumerate(VAL_DATA):
        predictions = MODEL(img, labels[:, :-1], lengths - 1)
        predictions = pack_padded_sequence(
            predictions, lengths - 1, batch_first=True)[0]
        labels = pack_padded_sequence(
            labels[:, 1:], lengths - 1, batch_first=True)[0]
        loss = MODEL.criterion(predictions, labels)
        error += loss.detach().item()
    WRITER.add_scalar('data/val_loss', error / len(VAL_DATA), epoch)
    '''

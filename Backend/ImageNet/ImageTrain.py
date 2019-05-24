'''Main method for training the network outlined in NeuralNetworks.py and Attention.py'''

# dependencies
import gc
import sys
import torch
from tensorboardX import SummaryWriter

# user defined modules
from HyperParameters import OPTIONS
from NeuralNetworks import Image2Caption
from DataLoader import TRAIN_LOADER as DATALOADER
from pack_padded_sequence import pack_padded_sequence

# load in hyper-parameters from python file
NUM_EPOCHS = OPTIONS['num_epochs']
ITR_SAVE = OPTIONS['itr_save']

# initialize model and loss function
MODEL = None

''' to continue training a previously saved model type: "python ImageTrain.py continue"
to start training a new model type: "python ImageTrain.py restart"'''

if sys.argv[1] == 'continue':
    MODEL = torch.load('img_embedding_model.pth')
elif sys.argv[1] == 'restart':
    MODEL = Image2Caption()
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

for epoch in range(NUM_EPOCHS):
    error = 0
    for i, (img, labels, lengths) in enumerate(DATALOADER):
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
        error += loss.detach().item()
        print('epoch {} of {} --- iteration {} of {}'.format(epoch + 1, NUM_EPOCHS,
                                                             i + 1, len(DATALOADER)),
              end='\r')
    torch.save(MODEL, 'img_embedding_model_{}.pth'.format(epoch))
    WRITER.add_scalar('data/train_loss', error / len(DATALOADER), epoch)

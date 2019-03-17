#dependencies
import torch
from tensorboardX import SummaryWriter
import gc
import sys

#user defined modules
from HyperParameters import options
from NeuralNetworks import Image2Caption
from DataLoader import train_loader as dataloader
from pack_padded_sequence import pack_padded_sequence


#modules used for testing and viewing
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


#load in hyper-parameters from python file
num_epochs = options['num_epochs']
batch_size = options['batch_size']
learning_rate = options['learning_rate']

#initialize model and loss function
model = None

if sys.argv[1] == 'continue':
    model = torch.load('img_embedding_model.pth')
#to start training a new model type: "python filename restart"
elif sys.argv[1] == 'restart':
    model = Image2Caption()
print('Note model parameters:\n{}'.format(model.parameters))

#create a logger
writer = SummaryWriter()
#writer.add_graph(model, (torch.randn(1,3,224,224), torch.randint(20,(1,1),dtype=torch.long),torch.randint(1,(1,1),dtype=torch.long)),verbose=True)

for epoch in range(num_epochs):
	error = 0
	gc.collect()
	for i,(img,labels,lengths) in enumerate(dataloader):
		model.optimizer.zero_grad()
		predictions = model(img,labels[:,:-1], lengths-1)
		predictions = pack_padded_sequence(predictions,lengths-1,batch_first=True)[0]
		labels = pack_padded_sequence(labels[:,1:],lengths-1,batch_first=True)[0]
		loss = model.criterion(predictions,labels)
		loss.backward()
		model.optimizer.step()
		error += loss.detach().item()
		if (i+1)%600 == 0:
			torch.save(model,'img_embedding_model.pth')
		print('epoch {} of {} --- iteration {} of {}'.format(epoch+1, num_epochs, i+1, len(dataloader)), end='\r')
	writer.add_scalar('data/train_loss', error/len(dataloader), epoch)
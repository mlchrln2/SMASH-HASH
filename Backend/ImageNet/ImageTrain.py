#dependencies
import torch
from tensorboardX import SummaryWriter
import gc
import sys

#user defined modules
from HyperParameters import options
from NeuralNetworks import Image2Caption
from DataLoader import dataloader

#modules used for testing and viewing
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

#create a logger
writer = SummaryWriter('ImageNetSummary/')

#load in hyper-parameters from python file
num_epochs = options['num_epochs']
batch_size = options['batch_size']
learning_rate = options['learning_rate']

#set up a method for drawing the images
to_img=ToPILImage()

#initialize model and loss function
model = None

if sys.argv[1] == 'continue':
    model = torch.load('img_embedding_model.sav')
#to start training a new model type: "python filename restart"
elif sys.argv[1] == 'restart':
    model = Image2Caption()
print('Note model parameters:\n{}'.format(model.parameters))
#writer.add_graph(model, (torch.randn(1,3,640,640), torch.zeros(1,17)))

for epoch in range(num_epochs):
	error = 0
	gc.collect()
	for i,(img,labels,lengths) in enumerate(dataloader):
		labels = labels[:,0,:].squeeze(1)
		predictions, summaries = model(img,labels)
		loss = model.criterion(predictions,labels)
		loss.backward()
		model.optimizer.step()
		error += loss.detach().item()
		print('epoch {} of {} --- iteration {} of {}'.format(epoch+1, num_epochs, i+1, len(dataloader)), end='\r')
	writer.add_scalar('data/train_loss', error/len(dataloader), epoch)
	torch.save(model,'img_embedding_model.pth')
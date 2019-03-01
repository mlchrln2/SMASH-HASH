#dependencies
import torch
from tensorboardX import SummaryWriter
import gc
import sys

#user defined modules
from HyperParameters import options
from NeuralNetworks import Image2Caption
from DataLoader import test_loader as dataloader

#modules used for testing and viewing
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from Vocabulary import idx2word

#load in hyper-parameters from python file
num_epochs = options['num_epochs']
batch_size = 1
learning_rate = options['learning_rate']

#set up a method for drawing the images
to_img=ToPILImage()

#initialize model and loss function
model = torch.load('img_embedding_model.pth')
print('Note model parameters:\n{}'.format(model.parameters))

#create a logger
writer = SummaryWriter('ImageNetSummary/')
writer.add_graph(model, (torch.randn(1,3,224,224), torch.randint(20,(1,20),dtype=torch.long)),vervose=True)

error = 0
gc.collect()
for i,(img,labels,lengths) in enumerate(dataloader):
	word_predictions = model.infer(img)
	sentence = [idx2word[word.item()] for word in word_predictions[0] if word.item() != 1 and word.item() != 0]
	sentence = ' '.join(sentence)
	print(sentence)
	img = to_img(img[0])
	fig = plt.figure(i)
	ax1 = fig.add_subplot(1,1,1)
	ax1.imshow(img)
	plt.title(sentence)
	plt.show()
	writer.add_figure('plot_{}'.format(i), fig, i, True)
	plt.close()
	print('iteration {} of {}'.format(i+1, len(dataloader)), end='\r')
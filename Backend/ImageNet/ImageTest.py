#dependencies
import torch
from tensorboardX import SummaryWriter
import gc

#user defined modules
from HyperParameters import options
from NeuralNetworks import Image2Caption
from DataLoader import test_loader as dataloader
from DataLoader import idx2word

#modules used for testing and viewing
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

#set up a method for drawing the images
to_img=ToPILImage()


def plot(img,frames,caption,i):
	fig = plt.figure(i,figsize=(2,1))
	img = to_img(img.squeeze(0))
	plt.title(caption)
	ax1 = fig.add_subplot(2,1,1)
	plt.imshow(img)
	ax2 = fig.add_subplot(2,1,2)
	img2 = to_img(frames)
	ax2.set_ylabel(frames.size(1))
	ax2.set_xlabel(frames.size(2))
	plt.imshow(img2)
	plt.show()
	writer.add_figure('plot_{}'.format(i), fig, i, True)
	plt.close()

#load in hyper-parameters from python file
num_epochs = options['num_epochs']
batch_size = 1
learning_rate = options['learning_rate']

#initialize model and loss function
model = torch.load('img_embedding_model.pth')
print('Note model parameters:\n{}'.format(model.parameters))

#create a logger
writer = SummaryWriter('ImageNetSummary/')
#writer.add_graph(model, (torch.randn(1,3,224,224), torch.randint(20,(1,20),dtype=torch.long)),vervose=True)

error = 0
gc.collect()
for i,(img,labels,lengths) in enumerate(dataloader):
	words, summaries, frames = model.infer(img)
	sentence = ' '.join([idx2word[str(word.item())].value.decode("utf-8") for word in words])
	plot(img, frames, sentence,i)
	print('iteration {} of {}'.format(i+1, len(dataloader)), end='\r')
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
import skimage.transform

#set up a method for drawing the images
to_img=ToPILImage()


def plot(img,alphas,caption,i):
	alphas = alphas.squeeze(0)
	width = alphas.size(0)
	fig = plt.figure(i,figsize=(10,width))
	ax1 = fig.add_subplot(2,1,1)
	image = to_img(img.squeeze(0))
	plt.imshow(image)
	for j in range(width):
		fig.add_subplot(2,width,j+width+1)
		plt.title(caption[j])
		plt.imshow(image)
		alpha = to_img(alphas[j].unsqueeze(0))
		plt.imshow(alpha,cmap='gray',alpha=.5)
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
writer = SummaryWriter()
#writer.add_graph(model, (torch.randn(1,3,224,224), torch.randint(20,(1,20),dtype=torch.long)),vervose=True)

error = 0
gc.collect()
for i,(image,img,labels,lengths) in enumerate(dataloader):
	words, summaries, alphas = model.infer(img)
	sentence = [idx2word[str(word.item())].value.decode("utf-8") for word in words]
	plot(image, alphas, sentence,i)
	print('iteration {} of {}'.format(i+1, len(dataloader)), end='\r')
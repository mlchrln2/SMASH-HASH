import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CocoCaptions
from torchvision.datasets import CocoDetection
import pathlib
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from HyperParameters import options
from NeuralNetworks import Autoencoder
from NeuralNetworks import Image2Caption
from NeuralNetworks import Attention
from Vocabulary import word2idx
from Vocabulary import idx2word
from tensorboardX import SummaryWriter
import gc
import sys

def collate(data):
	'''pads the data with 0's'''
	images, captions = zip(*data)
	length = torch.tensor([[item.size(1),item.size(2)] for item in images])
	(max_i,max_j),_ = torch.max(length,0)
	images = [nn.ZeroPad2d((0, max_j.item()-item.size(2), 0, max_i.item()-item.size(1)))
			(item) for item in images]
	images = torch.stack(images, 0)

	length = torch.tensor([[item.size(0),item.size(1)] for item in captions])
	(max_i,max_j),_ = torch.max(length,0)
	captions = [nn.ZeroPad2d((0, max_j.item()-item.size(1), 0, max_i.item()-item.size(0)))
			   (item) for item in captions]
	captions = torch.stack(captions, 0)	
	return images, captions

#create a logger
writer = SummaryWriter('ImageNetSummary/')

#load in hyper-parameters from python file
num_epochs = options['num_epochs']
batch_size = options['batch_size']
learning_rate = options['learning_rate']

#set up a method for drawing the images
to_img=ToPILImage()

#dataset location in memory
myPassport_dir = '/media/orlandomelchor/My Passport/datasets/coco-dataset'

class Word2Int(object):
	def __call__(self,sentences):
		max_sentence =  torch.max(torch.tensor([len(sentence.split(' ')) for sentence in sentences]))
		words = torch.tensor([[word2idx[sentence.split(' ')[i]] if i < len(sentence.split(' ')) 
							 else 0 for i in range(max_sentence)] for sentence in sentences]).long()
		return words


#transform = transforms.ToTensor()
transform = transforms.transforms.ToTensor()
target_transform = Word2Int()

#load data and captions in batches
dataset = CocoCaptions(root='{}/train2017/'.format(myPassport_dir), 
					   annFile='{}/annotations/captions_train2017.json'.format(myPassport_dir),
					   transform=transform,
					   target_transform=target_transform)
dataloader = DataLoader(dataset=dataset, 
						batch_size=batch_size,
						collate_fn=collate,
						shuffle=True,
						drop_last=True)

#initialize model and loss function
model = None

if sys.argv[1] == 'continue':
    model = torch.load('img_embedding_model.sav')
#to start training a new model type: "python filename restart"
elif sys.argv[1] == 'restart':
    model = Image2Caption()
print('Note model parameters:\n{}'.format(model.parameters))

for epoch in range(num_epochs):
	error = 0
	gc.collect()
	for i,(img,captions) in enumerate(dataloader):
		rand_caption = torch.randint(high=captions.size(1),size=(1,))
		captions = captions[:,rand_caption,:].squeeze(1)
		output_captions, input_captions = model(img,captions)
		loss = model.criterion(output_captions,input_captions)
		loss.backward()
		model.optimizer.step()
		error += loss.detach().item()
		print('epoch {} of {} --- iteration {} of {}'.format(epoch+1, num_epochs, i+1, len(dataloader)), end='\r')
		writer.add_scalar('data/train_loss', error/(i+1), epoch*len(dataloader)+i)
	torch.save(model,'img_embedding_model.sav')
'''
output = model(img)
loss = criterion(output, img)
model.optimizer.zero_grad()
loss.backward()
model.optimizer.step()
print('epoch [{}/{}], loss:{:.4f}'
	.format(epoch + 1, num_epochs, loss.data.item()))
if epoch % 10 == 0:
	pic = to_img(output.cpu().data)
	save_image(pic, './output/output_images/image_{}.png'.format(epoch))
'''

torch.save(model.state_dict(), './output/conv_autoencoder_state.pth')

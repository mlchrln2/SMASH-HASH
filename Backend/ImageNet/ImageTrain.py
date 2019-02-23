import torch
import torch.nn as nn
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
from NeuralNetworks import Attention
from tensorboardX import SummaryWriter
import gc
import sys

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

transform = transforms.Compose([
	transforms.Resize(size=(640,640)),
	transforms.ToTensor()
])
#transform = transforms.ToTensor()
#load data and captions in batches
dataset = CocoCaptions(root='{}/train2017/'.format(myPassport_dir), 
					   annFile='{}/annotations/captions_train2017.json'.format(myPassport_dir),
					   transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#initialize model and loss function
model = None

if sys.argv[1] == 'continue':
    model = torch.load('img_embedding_model.sav')
#to start training a new model type: "python filename restart"
elif sys.argv[1] == 'restart':
    model = Autoencoder()

for epoch in range(num_epochs):
	error = 0
	gc.collect()
	for i,data in enumerate(dataloader):
		img, _ = data
		output = model(img)
		loss = model.criterion(img,output)
		loss.backward()
		model.optimizer.step()
		error += loss.detach().item()
		print('epoch {} of {} --- iteration {} of {}'.format(epoch+1, num_epochs, i+1, len(dataloader)), end='\r')
	writer.add_scalar('data/train_loss', error/itr, epoch)
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

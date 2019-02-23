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

'''
load in hyper-parameters from python file
'''
num_epochs = options['num_epochs']
batch_size = options['batch_size']
learning_rate = options['learning_rate']

'''
set up a method for drawing the images
'''
to_img=ToPILImage()

'''
dataset location in memory
'''
myPassport_dir = '/media/orlandomelchor/My Passport/datasets/coco-dataset'

'''
load data and captions in batches
'''
dataset = CocoCaptions(root='{}/train2017/'.format(myPassport_dir), 
					   annFile='{}/annotations/captions_train2017.json'.format(myPassport_dir),
					   transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

'''
initialize model and loss function
'''
model = Autoencoder()
criterion = nn.MSELoss()

words = {}
itr = 0
for i,data in enumerate(dataloader):
	img, caption = data
	for cap in caption: 
		for word in cap[0].split(' '):
			if word not in words.keys():
				words[word] = itr
				itr += 1
	print('iteration {} of {}'.format(i,len(dataloader)),end='\r')
print(len(words))
print(words)
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

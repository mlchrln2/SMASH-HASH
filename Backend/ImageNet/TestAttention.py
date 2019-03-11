#dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

#user defined modules
from HyperParameters import options
from DataLoader import train_loader as dataloader
from Attention import LocalAttention2d

q = torch.tensor([[[[0.9700],
					[0.1700],
					[0.6700],
					[0.0600],
					[0.9300]],

				 [[0.7100],
					[0.6000],
					[0.4300],
					[0.1600],
					[0.2300]],

				 [[0.9700],
					[0.6900],
					[0.0800],
					[0.2100],
					[0.7000]],

				 [[0.4100],
					[0.2900],
					[0.7400],
					[0.3800],
					[0.8500]]],


				[[[0.7200],
					[0.0900],
					[0.5100],
					[0.7600],
					[0.4800]],

				 [[0.1900],
					[0.2700],
					[0.6300],
					[0.9600],
					[0.2800]],

				 [[0.3700],
					[0.2900],
					[0.2700],
					[0.3800],
					[0.9100]],

				 [[0.6400],
					[0.4000],
					[0.8200],
					[0.7100],
					[0.7300]]]])

p_t = torch.tensor([[[2.2091, 1.5382],
					 [3.2546, 0.6475],
					 [3.1470, 4.6770]],

					[[3.3531, 2.4096],
					 [1.4583, 1.8833],
					 [0.1498, 0.6317]]])

c_t = torch.tensor([[[0.0100],
					 [0.5500],
					 [0.8100]],

					[[0.8900],
					 [0.8800],
					 [0.4100]]])


W_a = torch.tensor([[0.0300]])

#set up a method for drawing the images
to_img=ToPILImage()

model = LocalAttention2d(query_size=3,
						 context_size=512,
						 align_size=10,
						 window=(25,25))
embedding = nn.Embedding(num_embeddings=29549,
						 embedding_dim=512)


q = q.permute(0,3,1,2)
#print(model.W_a.weight)
for i,(img,labels,lengths) in enumerate(dataloader):
	words = embedding(labels)
	model.optimizer.zero_grad()
	output = model(img, words)
	error = model.loss(output, torch.randn_like(output))
	out, (W_attn, loc) = model.infer(img,words)
	model.weight_viz(img[0],W_attn[0],loc[0],labels[0])
	error.backward()
	model.optimizer.step()
	#print(model.W_a.weight)
	break
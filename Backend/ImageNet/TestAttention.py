#dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

#user defined modules
from HyperParameters import options
from DataLoader import train_loader as dataloader
from Attention import LocalAttention2d

q = torch.tensor([[[[0.6200, 0.1500, 0.5400, 0.5100, 0.7800],
		          [0.6500, 0.7500, 0.5600, 0.9300, 0.7800],
		          [0.9400, 0.6300, 0.1100, 0.9600, 0.9700],
		          [0.0000, 0.3600, 0.8700, 0.3100, 0.7900]]],


		        [[[0.0200, 0.7500, 0.3000, 0.7400, 0.3100],
		          [0.0600, 0.3500, 0.0300, 0.2400, 0.5200],
		          [0.1700, 0.4500, 0.3200, 0.6000, 0.6500],
		          [0.7100, 0.7000, 0.3000, 0.4800, 0.3300]]]])


p_t = torch.tensor([[0.1682, 3.6395],
			        [3.3280, 1.1567]])

c_t = torch.tensor([[0.7000, 0.6100, 0.9500, 0.0100, 0.0700, 0.4700],
			        [0.2100, 0.8300, 0.6900, 0.7700, 0.2000, 0.5300]])



W_a = torch.tensor([[0.3000],
        [0.0700],
        [0.7700],
        [0.3300],
        [0.1000],
        [0.3800]])


#set up a method for drawing the images
to_img=ToPILImage()

model = LocalAttention2d(query_size=1,
						 context_size=6,
						 window=(3,3))

optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-4)
loss = nn.BCELoss()

model.W_a.weight.data = W_a
print(model.W_a.weight)
for i,(img,labels,lengths) in enumerate(dataloader):
	optimizer.zero_grad()
	output = model(q[0].unsqueeze(0), c_t[0].unsqueeze(0))
	print(output)
	error = loss(output, torch.randn_like(output))
	error.backward()
	optimizer.step()
	print(model.W_a.weight)
	break
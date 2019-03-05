import time
import torch
import torch.nn as nn

batches = 2
queries = 9
channels = 5
words = 3
word_size = 4
D = 3
S = queries

lin = nn.Linear(channels,word_size,bias=False)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=lin.parameters(), lr=1, weight_decay=1e-5)
optimizer.zero_grad()
#q = torch.arange(batches*queries*channels).view(batches,queries,channels).float()
#p_t = torch.tensor([[[0.11],[5.32]],[[1.48],[10]]])
#c_t = torch.arange(batches*words*word_size).view(batches,words,word_size).float()
q = torch.tensor([[[0.2600, 0.5600, 0.2900, 0.6700, 0.8100],
		         [0.5900, 0.4000, 0.5300, 0.6800, 0.5200],
		         [0.5500, 0.8800, 0.5500, 0.1600, 0.6200],
		         [0.5700, 0.6300, 0.2000, 0.6500, 0.2200],
		         [0.2000, 0.4200, 0.1100, 0.5900, 0.4600],
		         [0.3900, 0.5300, 0.5800, 0.2900, 0.9600],
		         [0.0600, 0.9000, 0.5800, 0.6100, 0.2200],
		         [0.4800, 0.3000, 0.1200, 0.8600, 0.1600],
		         [0.1400, 0.6600, 0.9900, 0.8200, 0.8000]],

		        [[0.6500, 0.9400, 0.2500, 0.5000, 0.2500],
		         [0.2200, 0.6000, 0.6200, 0.3400, 0.7700],
		         [0.0900, 0.5600, 0.2900, 0.4300, 0.9400],
		         [0.8000, 0.9500, 0.8100, 0.5400, 0.4600],
		         [0.5800, 0.7700, 0.3500, 0.0800, 0.7600],
		         [0.4600, 0.7600, 0.2800, 0.7800, 0.4100],
		         [0.4100, 0.5000, 0.2800, 0.6200, 0.7300],
		         [0.4000, 0.9500, 0.8900, 0.7700, 0.8100],
		         [0.7400, 0.5300, 0.6700, 0.2500, 0.2600]]])

p_t = torch.tensor([[[6.6600],
			         [1.1700],
			         [4.9500]],

			        [[7.5600],
			         [4.9500],
			         [3.2400]]])

c_t = torch.tensor([[[0.7400, 0.3400, 0.8000, 0.6600],
			         [0.9200, 0.0500, 0.1700, 0.7700],
			         [0.0900, 0.5000, 0.8300, 0.9700]],

			        [[0.5000, 0.0400, 0.0900, 0.2300],
			         [0.2400, 0.2500, 0.9200, 0.5900],
			         [0.3700, 0.3900, 0.5600, 0.1200]]])

lin.weight.data = torch.tensor([[ 0.0823, -0.1433,  0.3603,  0.3888, -0.0696],
						        [ 0.1195,  0.4177,  0.0176,  0.4141, -0.4175],
						        [ 0.3725, -0.0063, -0.0640,  0.2831, -0.0994],
						        [-0.1453, -0.1733, -0.0267,  0.2782,  0.3131]])

t0 = time.time()
q_new = nn.ConstantPad2d((0,0,1,0),float('nan'))(q)
q_new = torch.stack([q_new.gather(1,(torch.clamp((p_t+idx+1).long().repeat(1,1,q_new.size(2)),min=0,max=q_new.size(1))%(q_new.size(1)))) for idx in range(-D,D+1)],dim=2)
temp1 = torch.isnan(q_new[:,:,:,0])
q_new_1 = q_new.clone()
q_new_1[temp1] = 0
W_q_new = lin(q_new_1)
W_q_new[temp1] = float('nan')
a_t = torch.bmm(W_q_new.view(-1,W_q_new.size(2),W_q_new.size(3)),c_t.view(-1,c_t.size(2),1)).view(W_q_new.size(0),W_q_new.size(1),W_q_new.size(2))

temp2 = torch.isnan(a_t)
a_t = a_t.clone()
a_t[temp2] = -float('inf')
softmax = nn.Softmax(2)
prob = softmax(a_t).unsqueeze(3)
exponent = torch.stack([-2*pow((idx + p_t.long().float() - p_t)/D,2) for idx in range(-D,D+1)], dim=2)
prob = prob*torch.exp(exponent)
out = (prob*q_new_1).sum(2)
error = loss(out,torch.randn_like(out))
error.backward()
optimizer.step()

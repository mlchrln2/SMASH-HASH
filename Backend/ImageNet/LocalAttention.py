#dependencies
import torch
import torch.nn as nn

#user defined modules
from HyperParameters import options
from DataLoader import test_loader as dataloader
'''
General Soft-Attention Model
'''
class Attention(nn.Module):
	'''
		Attention parameters:
			c_size: context size
			h_size: query size
		Inputs:
			c_t: context of shape (N,1,c_size)
			q  : query of shape   (N,S,h_size)
		Formulation:
			score = c_t^T * W_a * q_i
			alpha_ti = exp(score) / sum_{i=1}^{n}{exp(score)}
			s_t = sum_{i=1}^{n}{alpha_ti * q_i}
		General idea:
			create a score for the current context at time t for every query i
			use the scores to create importance probabilites for every query i
			save the importance probabilities as the attention weights
			scale each query i by its importance
			sum the weighted queries together to produce the summary s_t
	'''
	def __init__(self, query_size, context_size):
		super(Attention,self).__init__()
		self.q_size = query_size
		self.c_size = context_size
		self.W_a = nn.Linear(in_features=self.q_size, 
							 out_features=self.c_size, 
							 bias=False)
		self.softmax = nn.Softmax(1)
	def forward(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), 1, self.c_size)
		W_attn = self.softmax(self.score(q,c_t))
		alpha_t = W_attn.transpose(1,2)
		s_t = torch.bmm(alpha_t,q)
		return s_t, W_attn
	def score(self,q,c_t):
		return torch.bmm(self.W_a(q),c_t.transpose(1,2))
	def align(q,c_t):
		W_attn = self.softmax(self.score(q,c_t))
		return W_attn

class LocalAttention(nn.Module):
	def __init__(self, query_size, context_size, alignment_size):
		super(LocalAttention,self).__init__()
		self.q_size = query_size
		self.c_size = context_size
		self.p_size = alignment_size
		self.W_a = nn.Linear(in_features=self.q_size,
							 out_features=self.c_size,
							 bias=False)
		self.W_p = nn.Linear(in_features=self.c_size,
							 out_features=self.p_size,
							 bias=False)
		self.V_p = nn.Linear(in_features=self.p_size,
							 out_features=1,
							 bias=False)
		self.softmax = nn.Softmax(2)
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.D = 3
		self.loss = nn.MSELoss()
		self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1, weight_decay=1e-5)
	def forward(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), 1, self.c_size)
		p_t = self.predictive_alignment(q.size(2), c_t)
		q = nn.ConstantPad1d((1,0),float('nan'))(q)
		q = q.transpose(1,2)
		s = torch.clamp(torch.stack([p_t.squeeze(2).long()+idx+1 for idx in range(-self.D,self.D+1)],dim=2),min=0,max=q.size(1))%q.size(1)
		q = torch.stack([batch[idx] for batch,idx in zip(q,s)])
		nan_loc = torch.isnan(q[...,0])
		q[nan_loc] = 0
		W_attn = self.align(q,c_t,nan_loc)*torch.exp(-2*torch.pow((torch.clamp(s.float()-1,min=0)-p_t)/self.D,2))
		out = (W_attn.unsqueeze(-1)*q).sum(2)
		return out
	def predictive_alignment(self, S, c_t):
		p_t = S * self.sigmoid(self.V_p(self.tanh(self.W_p(c_t))))
		return p_t
	def align(self,q,c_t,nan_loc):
		a_t = self.score(q,c_t)
		a_t[nan_loc] = -float('inf')
		W_attn = self.softmax(a_t)
		return W_attn
	def score(self,q,c_t):
		Wa = self.W_a(q)
		a_t = torch.bmm(Wa.view(-1,Wa.size(2),Wa.size(3)),c_t.view(-1,c_t.size(2),1)).view(Wa.size(0),Wa.size(1),Wa.size(2))
		return a_t

model = LocalAttention(5, 4,100)

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

c_t = torch.tensor([[[0.7400, 0.3400, 0.8000, 0.6600],
			         [0.9200, 0.0500, 0.1700, 0.7700],
			         [0.0900, 0.5000, 0.8300, 0.9700]],

			        [[0.5000, 0.0400, 0.0900, 0.2300],
			         [0.2400, 0.2500, 0.9200, 0.5900],
			         [0.3700, 0.3900, 0.5600, 0.1200]]])

p_t = torch.tensor([[[6.6600],
			         [1.1700],
			         [4.9500]],

			        [[7.5600],
			         [4.9500],
			         [3.2400]]])


model.W_a.weight.data = torch.tensor([[ 0.0823, -0.1433,  0.3603,  0.3888, -0.0696],
							        [ 0.1195,  0.4177,  0.0176,  0.4141, -0.4175],
							        [ 0.3725, -0.0063, -0.0640,  0.2831, -0.0994],
							        [-0.1453, -0.1733, -0.0267, 0.2782, 0.3131]])


for i,(img,labels,lengths) in enumerate(dataloader):
	model.optimizer.zero_grad()
	q = q.transpose(1,2)
	output = model(q, c_t)
	error = model.loss(output, torch.randn_like(output))
	print(model.W_a.weight)
	error.backward()
	model.optimizer.step()
	print(model.W_a.weight)
	break
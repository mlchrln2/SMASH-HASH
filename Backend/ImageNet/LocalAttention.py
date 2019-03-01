#dependencies
import torch
import torch.nn as nn

#user defined modules
from HyperParameters import options
from DataLoader import dataloader
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
	def __init__(self, query_size, context_size):
		super(LocalAttention,self).__init__()
		self.q_size = query_size
		self.c_size = context_size
		self.p_size = 100
		self.W_a = nn.Linear(in_features=self.q_size,
							 out_features=self.c_size,
							 bias=False)
		self.W_p = nn.Linear(in_features=self.c_size,
							 out_features=self.p_size,
							 bias=False)
		self.V_p = nn.Linear(in_features=self.p_size,
							 out_features=1,
							 bias=False)
		self.softmax = nn.Softmax(1)
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.window_length = 2
		self.stan_dev = self.window_length/2
	def forward(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), 1, self.c_size)
		S = q.size(1)
		p_t = self.predictive_alignment(S, c_t)
		win_low = (p_t-self.window_length).long()
		win_high =  (p_t+self.window_length).long()
		for low_arr, high_arr in zip(win_low, win_high):
			for low,high in zip(low_arr,high_arr):
				print(torch.arange(low.item(),high.item()))
		window = torch.tensor([[[elem for elem in range(low.item(),high.item())] 
								for low,high in zip(low_arr,high_arr)]
								for low_arr,high_arr in zip(win_low,win_high)])
		window = torch.clamp(torch.clamp(window,max=S),min=0)
		q = q[:,window]
		print(q.size(),'0')
		a_t = self.align(q,c_t)*torch.exp(-(s-p_t)/(2*torch.pow(self.stan_dev,2)))
		s_t = torch.bmm(a_t,q)
		return s_t
	def predictive_alignment(self, S, c_t):
		p_t = S * self.sigmoid(self.V_p(self.tanh(self.W_p(c_t))))
		return p_t
	def align(self,q,c_t):
		W_attn = self.softmax(self.score(q,c_t))
		return W_attn
	def score(self,q,c_t):
		print(q.size(),'1')
		print(c_t.size(),'2')
		print(self.W_a(q).size(),'3')
		return torch.bmm(self.W_a(q),c_t.transpose(1,2))

model = LocalAttention(3, 512)
for _,_,_ in dataloader:
	output = model(torch.randn(2,10,3),torch.randn(2,2,512))
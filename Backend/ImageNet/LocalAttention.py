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
		self.softmax = nn.Softmax(2)
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.D = 50
		self.loss = nn.MSELoss()
		self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1, weight_decay=1e-5)
	def forward(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), 1, self.c_size)
		S = q.size(1)
		p_t = self.predictive_alignment(S, c_t)
		p_t = p_t.clone()
		q = nn.ConstantPad2d((0,0,1,0),float('nan'))(q)
		q = torch.stack([q.gather(1,(torch.clamp((p_t+idx+1).long().repeat(1,1,q.size(2)),
						 min=0,max=q.size(1))%q.size(1))) for idx in range(-self.D,self.D+1)],dim=2)
		W_attn,q1 = self.align(q,c_t)
		exponent = torch.stack([-2*pow((idx + p_t.long().float() - p_t)/self.D,2) 
								for idx in range(-self.D,self.D+1)], dim=2)
		W_attn = W_attn*torch.exp(exponent)
		s_t = (W_attn*q1).sum(2)
		return s_t
	def predictive_alignment(self, S, c_t):
		p_t = S * self.sigmoid(self.V_p(self.tanh(self.W_p(c_t))))
		return p_t
	def align(self,q,c_t):
		s_out, q1 = self.score(q,c_t)
		W_attn = self.softmax(s_out).unsqueeze(3)
		return W_attn,q1
	def score(self,q_new,c_t):
		temp1 = torch.isnan(q_new[:,:,:,0])
		q_new_1 = q_new.clone()
		q_new_1[temp1] = 0
		W_q_new = self.W_a(q_new_1)
		W_q_new[temp1] = float('nan')
		a_t = torch.bmm(W_q_new.view(-1,W_q_new.size(2),W_q_new.size(3)),c_t.view(-1,c_t.size(2),1)).view(W_q_new.size(0),W_q_new.size(1),W_q_new.size(2))
		temp2 = torch.isnan(a_t)
		a_t = a_t.clone()
		a_t[temp2] = -float('inf')
		return a_t, q_new_1

model = LocalAttention(3, 512)
for i,(img,labels,lengths) in enumerate(dataloader):
	model.optimizer.zero_grad()
	output = model(torch.randn(8,224*224,3), torch.randn(8,20,512))
	error = model.loss(output, torch.randn_like(output))
	error.backward()
	model.optimizer.step()
	print(model.W_a.weight)
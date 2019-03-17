#dependencies
import torch
import torch.nn as nn

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
		W_attn = self.align(q,c_t)
		alpha_t = W_attn.transpose(1,2)
		s_t = torch.bmm(alpha_t,q)
		return s_t, W_attn
	def score(self,q,c_t):
		return torch.bmm(self.W_a(q),c_t.transpose(1,2))
	def align(q,c_t):
		W_attn = self.softmax(self.score(q,c_t))
		return W_attn

class LocalAttention1d(nn.Module):
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

class LocalAttention2d(nn.Module):
	def __init__(self, query_size, context_size, window=(1,1)):
		super(LocalAttention2d,self).__init__()
		self.q_size = query_size
		self.c_size = context_size
		self.W_a = nn.Linear(in_features=self.q_size,
							 out_features=self.c_size,
							 bias=False)
		self.W_p = nn.Linear(in_features=self.c_size,
							 out_features=2,
							 bias=False)
		self.softmax = nn.Softmax(1)
		self.sigmoid = nn.Sigmoid()
		self.R = window[0]
		self.C = window[1]
	def forward(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), self.c_size)
		p_t = self.predictive_alignment(q.size(2), c_t)
		q = nn.ConstantPad2d((1,0,1,0),float('nan'))(q)
		rows, cols = q.size(2),q.size(3)
		q = q.view(q.size(0),q.size(1),-1)
		q = q.transpose(1,2)
		r = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch-(self.R//2)+1,p_batch+(self.R+1)//2+1) 
			for p_batch in torch.round(p_t[:,0]).long()]),min=0,max=rows),rows)
		c = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch-(self.R//2)+1,p_batch+(self.R+1)//2+1) 
			for p_batch in torch.round(p_t[:,1]).long()]),min=0,max=rows),rows)
		s = torch.stack([r[:,i]*cols+c[:,j]
						for i in range(r.size(-1)) 
						for j in range(c.size(-1))],dim=1)
		q = torch.stack([batch[idx] for batch,idx in zip(q,s)])
		nan_loc = torch.isnan(q[...,0])
		q[nan_loc] = 0
		rexp = 2*torch.pow((torch.clamp(r-1,min=0).float()-p_t[:,0].unsqueeze(-1))/(self.R//2),2)
		cexp = 2*torch.pow((torch.clamp(c-1,min=0).float()-p_t[:,1].unsqueeze(-1))/(self.C//2),2)
		shift = torch.stack([rexp[...,i]+cexp[...,j]
							for i in range(rexp.size(-1)) 
							for j in range(cexp.size(-1))],dim=1)
		W_attn = self.align(q,c_t,nan_loc,shift)
		out = torch.bmm(W_attn.transpose(1,2),q).squeeze(1)
		return out
	def infer(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), self.c_size)
		p_t = self.predictive_alignment(q.size(2), c_t)
		q = nn.ConstantPad2d((1,0,1,0),float('nan'))(q)
		img = q
		rows, cols = q.size(2),q.size(3)
		q = q.view(q.size(0),q.size(1),-1)
		q = q.transpose(1,2)
		r = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch-(self.R//2)+1,p_batch+(self.R+1)//2+1) 
			for p_batch in torch.round(p_t[:,0]).long()]),min=0,max=rows),rows)
		c = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch-(self.R//2)+1,p_batch+(self.R+1)//2+1) 
			for p_batch in torch.round(p_t[:,1]).long()]),min=0,max=rows),rows)
		s = torch.stack([r[:,i]*cols+c[:,j]
						for i in range(r.size(-1)) 
						for j in range(c.size(-1))],dim=1)
		q = torch.stack([batch[idx] for batch,idx in zip(q,s)])
		nan_loc = torch.isnan(q[...,0])
		q[nan_loc] = 0
		rexp = 2*torch.pow((torch.clamp(r-1,min=0).float()-p_t[:,0].unsqueeze(-1))/(self.R//2),2)
		cexp = 2*torch.pow((torch.clamp(c-1,min=0).float()-p_t[:,1].unsqueeze(-1))/(self.C//2),2)
		shift = torch.stack([rexp[...,i]+cexp[...,j]
							for i in range(rexp.size(-1)) 
							for j in range(cexp.size(-1))],dim=1)
		W_attn = self.align(q,c_t,nan_loc,shift)
		out = torch.bmm(W_attn.transpose(1,2),q).squeeze(1)
		W_attn = W_attn.view(W_attn.size(0),self.R,self.C)
		s = s.view(s.size(0),self.R,self.C)
		s = torch.stack([s//cols, torch.fmod(s,cols)],dim=3)
		alpha = self.alpha_frames(img, W_attn, s)
		return out, alpha
	def predictive_alignment(self, S, c_t):
		p_t = S * self.sigmoid(self.W_p(c_t))
		return p_t
	def align(self,q,c_t,nan_loc,shift):
		a_t = self.score(q,c_t) - shift.unsqueeze(2)
		a_t[nan_loc] = -float('inf')
		W_attn = self.softmax(a_t)
		return W_attn
	def score(self,q,c_t):
		a_t = torch.bmm(self.W_a(q),c_t.unsqueeze(2))
		return a_t
	def alpha_frames(self,q,W_attn,loc):
		W_attn, loc = W_attn.squeeze(0), loc.squeeze(0)
		q = q.permute(0,2,3,1)
		alpha_matrix = torch.zeros(q.size(0),q.size(1),q.size(2))
		alpha_matrix[:,loc[...,0],loc[...,1]] = W_attn[0]
		return alpha_matrix[:,1:,1:]
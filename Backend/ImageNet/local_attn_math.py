import time
import torch
import torch.nn as nn
class LocalAttention(nn.Module):
	def __init__(self):
		super(LocalAttention,self).__init__()
		self.batches = 2
		self.queries = 9
		self.channels = 5
		self.words = 3
		self.word_size = 4
		self.D = 3
		self.p_size = 100
		self.W_a = nn.Linear(self.channels,self.word_size,bias=False)
		self.W_p = nn.Linear(in_features=self.word_size,
							 out_features=self.p_size,
							 bias=False)
		self.V_p = nn.Linear(in_features=self.p_size,
							 out_features=1,
							 bias=False)
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.loss = nn.MSELoss()
		self.softmax = nn.Softmax(2)
		self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1, weight_decay=1e-5)
	def forward(self,q,c_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), 1, self.c_size)
		S = q.size(1)
		p_t = self.predictive_alignment(S, c_t)
		s = torch.stack([p_t.squeeze(2).long()+idx for idx in range(-self.D,self.D+1)],dim=2)
		q = nn.ConstantPad2d((0,0,1,0),float('nan'))(q)
		q = torch.stack([batch[idx] for batch,idx in zip(q,torch.clamp(s+1,min=0,max=q.size(1))%q.size(1))])
		prob, q = self.align(q, c_t)
		prob = prob*torch.exp(-2*torch.pow((s.float()-p_t)/self.D,2))
		out = (prob.unsqueeze(-1)*q).sum(2)
		return out
	def align(self,q,c_t):		
		a_t, q = self.score(q,c_t)
		prob = self.softmax(a_t)
		return prob, q
	def score(self,q,c_t):
		temp1 = torch.isnan(q[:,:,:,0])
		q = q.clone()
		q[temp1] = 0
		W_q_new = self.W_a(q)
		W_q_new[temp1] = float('nan')
		a_t = torch.bmm(W_q_new.view(-1,W_q_new.size(2),W_q_new.size(3)),c_t.view(-1,c_t.size(2),1)).view(W_q_new.size(0),W_q_new.size(1),W_q_new.size(2))
		temp2 = torch.isnan(a_t)
		a_t = a_t.clone()
		a_t[temp2] = -float('inf')
		return a_t, q
	def predictive_alignment(self, S, c_t):
		p_t = S * self.sigmoid(self.V_p(self.tanh(self.W_p(c_t))))
		return p_t


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
model = LocalAttention()
output = model(q,c_t)
error = model.loss(output,torch.randn_like(output))
print(model.W_a.weight)
error.backward()
model.optimizer.step()
print(model.W_a.weight)

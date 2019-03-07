#dependencies
import torch
import torch.nn as nn

#user defined modules
from HyperParameters import options
from DataLoader import test_loader as dataloader

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


class LocalAttention2d(nn.Module):
	def __init__(self, query_size, context_size, alignment_size):
		super(LocalAttention2d,self).__init__()
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
							 out_features=2,
							 bias=False)
		self.softmax = nn.Softmax(2)
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.R = 1
		self.C = 2
		self.loss = nn.MSELoss()
		self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1, weight_decay=1e-5)
	def forward(self,q,c_t,p_t):
		if c_t is None:
			c_t = q.new_zeros(q.size(0), 1, self.c_size)
		#p_t = self.predictive_alignment(q.size(2), c_t)
		q = q.permute(0,3,1,2)
		q = nn.ConstantPad2d((1,0,1,0),float('nan'))(q)
		rows, cols = q.size(2),q.size(3)
		r = torch.clamp(torch.stack([p_t[...,0].long()+row+1 for row in range(-self.R,self.R+1)],dim=2),min=0,max=rows)%rows
		c = torch.clamp(torch.stack([p_t[...,1].long()+col+1 for col in range(-self.C,self.C+1)],dim=2),min=0,max=cols)%cols
		indexes = torch.stack([r[...,i]*cols+c[...,j] for i in range(r.size(-1)) for j in range(c.size(-1))],dim=2)
		q = q.view(q.size(0),q.size(1),-1)
		q = q.transpose(1,2)
		q = torch.stack([batch[idx] for batch,idx in zip(q,indexes)])
		nan_loc = torch.isnan(q[...,0])
		q[nan_loc] = 0
		rexp = torch.exp(-2*torch.pow((torch.clamp(r.float()-1,min=0)-p_t[...,0].unsqueeze(-1))/self.R,2))
		cexp = torch.exp(-2*torch.pow((torch.clamp(c.float()-1,min=0)-p_t[...,1].unsqueeze(-1))/self.C,2))
		exp = torch.stack([rexp[...,i]*cexp[...,j] for i in range(rexp.size(-1)) for j in range(cexp.size(-1))],dim=2)
		W_attn = self.align(q,c_t,nan_loc)*exp
		out = (W_attn.unsqueeze(-1)*q).sum(2)
		return out
	def predictive_alignment(self, S, c_t):
		p_t = S * self.sigmoid(self.V_p(self.tanh(self.W_p(c_t))))
		return p_t
	def align(self,q,c_t,nan_loc):
		a_t = self.score(q,c_t)
		a_t[nan_loc] = -float('inf')
		print(a_t)
		W_attn = self.softmax(a_t)
		return W_attn
	def score(self,q,c_t):
		Wa = self.W_a(q)
		a_t = torch.bmm(Wa.view(-1,Wa.size(2),Wa.size(3)),c_t.view(-1,c_t.size(2),1)).view(Wa.size(0),Wa.size(1),Wa.size(2))
		return a_t

model = LocalAttention2d(1,1,100)

for i,(img,labels,lengths) in enumerate(dataloader):
	model.optimizer.zero_grad()
	output = model(q, c_t, p_t)
	error = model.loss(output, torch.randn_like(output))
	error.backward()
	model.optimizer.step()
	break
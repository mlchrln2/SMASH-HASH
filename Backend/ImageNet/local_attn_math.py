import time
import torch
import torch.nn as nn

batches = 2
queries = 10
channels = 3
words = 2
word_size = 3

q = torch.arange(60).view(batches,queries,channels).float()
p_t = torch.tensor([[[0.11],[5.32]],[[1.48],[10]]]).float()
c_t = torch.arange(2*2*3).view(batches,words,word_size).float()
S = 10
D = 2

lin = nn.Linear(q.size(2),c_t.size(2))
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=lin.parameters(), lr=1e-5, weight_decay=1e-5)

q_new = nn.ConstantPad2d((0,0,1,0),float('nan'))(q)
q_new = torch.stack([q_new.gather(1,(torch.clamp((p_t+idx+1).long().repeat(1,1,q_new.size(2)),min=0,max=q_new.size(1))%(q_new.size(1)))) for idx in range(-D,D+1)],dim=2)
temp1 = torch.isnan(q_new)
q_new_1 = q_new.clone()
q_new_1[temp1] = 0
W_q_new = lin(q_new_1)
a_t = torch.bmm(W_q_new.view(-1,W_q_new.size(2),W_q_new.size(3)),c_t.view(-1,c_t.size(2),1)).view(W_q_new.size(0),W_q_new.size(1),W_q_new.size(2))
temp2 = torch.isnan(a_t)
a_t = a_t.clone()
a_t[temp2] = -float('inf')
softmax = nn.Softmax(2)
prob = softmax(a_t).unsqueeze(3)
out = (prob*q_new_1).sum(2)

error = loss(out,torch.randn_like(out))
error.backward()
optimizer.step()

lin.weight
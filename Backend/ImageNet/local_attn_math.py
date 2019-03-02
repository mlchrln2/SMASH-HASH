import time
import torch
import torch.nn as nn

q = torch.arange(60).view(2,10,3).float()
S = 10
D = 2
p_t_rand = torch.randn(2,2,1)*S
p_t = torch.tensor([[[0.11],[5.32]],[[1.48],[10]]]).float()
c_t = torch.arange(2*2*3).view(2,2,3).float()

q_new = nn.ConstantPad2d((0,0,1,0),-760)(q)
W_q_new = torch.stack([q_new.gather(1,(torch.clamp((p_t+idx+1).long().repeat(1,1,q_new.size(2)),min=0,max=q_new.size(1))%(q_new.size(1)))) for idx in range(-D,D+1)],dim=2)
a_t = torch.bmm(W_q_new.view(-1,W_q_new.size(2),W_q_new.size(3)),c_t.view(-1,c_t.size(2),1)).view(W_q_new.size(0),W_q_new.size(1),W_q_new.size(2))

softmax = nn.Softmax(2)
prob = softmax(a_t).unsqueeze(3)
out = (prob*W_q_new).sum(2)


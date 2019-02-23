import torch
import torch.nn as nn
from HyperParameters import options

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
            h_i: query of shape   (N,S,h_size)
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
    def __init__(self, q_size, c_size):
        super(Attention,self).__init__()
        self.q_size = q_size
        self.c_size = c_size
        self.W_a = nn.Linear(self.q_size, self.c_size, bias=False)
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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.num_epochs = options['n_batches']
        self.LR = options["learning_rate"]
        self.batch_size = options["batch_size"]
        self.hidden_size = options['hidden_neurons']

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, (2,2), stride=2, padding=1),
            nn.Tanh()
        )

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR,
                                          weight_decay=1e-5)
    def forward(self, x):
    	print(x.size())
    	x = self.encoder(x)
    	print(x.size())
    	x = self.decoder(x)
    	print(x.size())
    	return x

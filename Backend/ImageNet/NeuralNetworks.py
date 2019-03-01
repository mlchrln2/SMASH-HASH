#dependencies
import torch
import torch.nn as nn
import torchvision.models as models

#user defined modules
from HyperParameters import options

class Image2Caption(nn.Module):
	def __init__(self):
		super(Image2Caption, self).__init__()
		self.channel_size = options['channel_size']
		self.embed_size = options['embed_size']
		self.vocab_size = options['vocab_size']
		self.max_len = options['max_len']
		self.bidirectional = options['bidirectional']
		self.num_layers = options['num_layers']
		self.LR = options['learning_rate']
		self.hidden_size = options['hidden_size']
		self.image_encoder = ImageEncoder(channel_size=self.channel_size)
		self.word_encoder = WordEncoder(vocab_size=self.vocab_size, 
									    embed_size=self.embed_size,
									    hidden_size=self.hidden_size,
									    bidirectional=self.bidirectional, 
									    num_layers=self.num_layers)
		self.caption_decoder = CaptionDecoder(hidden_size=self.hidden_size,
											  channel_size=self.channel_size,
									          vocab_size=self.vocab_size, 
									          max_len=self.max_len,
									          bidirectional=self.bidirectional,
									          num_layers=self.num_layers)
		self.optimizer = torch.optim.Adam(params=self.parameters(), 
										  lr=self.LR,
										  weight_decay=1e-5)
		self.criterion = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(1)
	def forward(self, images, captions):
		features = self.image_encoder(images)
		captions = self.word_encoder(captions)
		captions, summaries = self.caption_decoder(features,captions)
		captions = self.softmax(captions.transpose(1,2))
		return captions, summaries
	def infer(self, image):
		features = self.image_encoder(image)
		caption = self.word_encoder(torch.zeros((features.size(0),1),dtype=torch.long))
		words = self.caption_decoder.infer(features, caption)
		return words

class ImageEncoder(nn.Module):
	def __init__(self, channel_size):
		super(ImageEncoder,self).__init__()
		self.channel_size = channel_size
		pretrained_net = models.vgg16(pretrained=True).features
		modules = list(pretrained_net.children())[:29]
		self.pretrained_net = nn.Sequential(*modules)
		self.out_channel = nn.Linear(in_features=modules[-1].in_channels,
									 out_features=self.channel_size)
	def forward(self, x):
		with torch.no_grad():
			x = self.pretrained_net(x)
		x = x.view(x.size(0),x.size(1),-1)
		x = x.transpose(1,2)
		x = self.out_channel(x)
		return x

class WordEncoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, bidirectional, num_layers):
		super(WordEncoder, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		self.num_layers = num_layers
		self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
									  embedding_dim=self.embed_size)
		self.rnn = nn.GRU(input_size=self.embed_size,
						  hidden_size=self.hidden_size, 
						  bidirectional=self.bidirectional,
						  num_layers=self.num_layers,
						  batch_first=True)
	def forward(self,captions):
		embed = self.embedding(captions)
		embed,hn = self.rnn(embed,None)
		return embed

class CaptionDecoder(nn.Module):
	def __init__(self, hidden_size, channel_size, vocab_size, max_len, bidirectional, num_layers):
		super(CaptionDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.channel_size = channel_size
		self.vocab_size = vocab_size
		self.max_len = max_len
		self.bidirectional = bidirectional
		self.bi_factor = (2 if self.bidirectional else 1)
		self.num_layers = num_layers
		self.attention = Attention(query_size=self.channel_size,
								   context_size=self.hidden_size*self.bi_factor)
		self.rnn = nn.GRU(input_size=self.channel_size,
						  hidden_size=self.hidden_size, 
						  bidirectional=self.bidirectional,
						  num_layers=self.num_layers,
						  batch_first=True)
		self.word_decoder = WordDecoder(self.hidden_size*self.bi_factor, 
										self.vocab_size)
	def forward(self,x,hn):
		x,x_weights = self.attention(x,hn)
		hn = hn[:,0]
		hn = hn.view(-1,self.bi_factor,self.hidden_size)
		x,hn = self.rnn(x,hn.transpose(0,1))
		hn = hn.transpose(0,1).contiguous()
		summaries = hn.view(-1,self.hidden_size*self.bi_factor)
		x = self.word_decoder(x)
		return x, summaries
	def infer(self,features,caption):
		words = torch.ones(features.size(0),self.max_len,dtype=torch.long)
		for i in range(self.max_len):
			x,x_weights = self.attention(features,caption)
			caption = caption.contiguous().view(-1,self.bi_factor,self.hidden_size)
			caption = caption.transpose(0,1)
			caption,hn = self.rnn(x,caption)
			word_vec = self.word_decoder(caption).squeeze(1)
			idxs = torch.argmax(word_vec,1)
			words[:,i] = idxs
			if idxs.item() == 1:
				break
		return words

'''
General Soft-Attention Model
'''
class Attention(nn.Module):
	'''
		Attention parameters:
			q_size: query size
			c_size: context size
		Inputs:
			q  : query of shape   (N,S,q_size)
			c_t: context of shape (N,1,c_size)
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
		a = torch.bmm(self.W_a(q),c_t.transpose(1,2))
		return a

class WordDecoder(nn.Module):
	def __init__(self, hidden_size, vocab_size):
		super(WordDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.decoder = nn.Linear(in_features=self.hidden_size, 
								 out_features=self.vocab_size)
	def forward(self, embed):
		captions = self.decoder(embed)
		return captions
	def infer(self, embed):
		embed = self.decoder(embed)
		idx = torch.argmax(embed)
		return idx

'''
Sample autoencoder network for testing on MNIST 
'''
class MNISTEncoder(nn.Module):
	def __init__(self):
		super(MNISTEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, 3, stride=3, padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(16, 8, 3, stride=2, padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=1)
		)
	def forward(self, x):
		x = self.encoder(x)
		return x

class MNISTDecoder(nn.Module):
	def __init__(self):
		super(MNISTDecoder, self).__init__()
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 16, 3, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
			nn.Tanh()
		)
	def forward(self, x):
		x = self.decoder(x)
		return x

class MNISTAutoencoder(nn.Module):
	def __init__(self):
		super(MNISTAutoencoder, self).__init__()
		self.LR = options['learning_rate']
		self.encoder = MNISTEncoder()
		self.decoder = MNISTDecoder()
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR,
										  weight_decay=1e-5)
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x
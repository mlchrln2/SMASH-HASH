import torch
import torch.nn as nn
from HyperParameters import options
import torchvision.models as models
from Vocabulary import word2idx
from Vocabulary import idx2word
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
			nn.MaxPool2d(2, stride=1),
			nn.Conv2d(8, 4, 3,stride=2,padding=1),
			nn.ReLU(True)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(4, 8, 1, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(8, 16, 3, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(8, 3, (2,2), stride=2, padding=1),
			nn.Tanh()
		)
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR,
										  weight_decay=1e-5)
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class ImageEncoder(nn.Module):
	def __init__(self):
		super(ImageEncoder,self).__init__()
		self.batches = options["batch_size"]
		self.channels = options['output_channels']
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 12, 3, stride=3, padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(12, 24, 5, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=1)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(24, 48, 5, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=1)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(48, self.channels, 3, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=1)
		)
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(self.batches,self.channels,-1)
		x = x.transpose(1,2)
		return x

class WordDecoder(nn.Module):
	def __init__(self, channels, vocab_size, pretrained=False):
		super(WordDecoder, self).__init__()
		self.channels = channels
		self.vocab_size = vocab_size
		self.decoder = nn.Linear(self.channels, self.vocab_size)
	def forward(self, x):
		x = self.decoder(x)
		return x
	def infer(self, x):
		x = self.decoder(x)
		idx = torch.argmax(x)
		return idx


class CaptionDecoder(nn.Module):
	def __init__(self):
		super(CaptionDecoder, self).__init__()
		self.batches = options["batch_size"]
		self.channels = options['output_channels']
		self.vocab_size = options['vocab_size']
		self.channels = options['output_channels']
		self.vocab_size = options['vocab_size']
		self.word_decoder = WordDecoder(self.channels, self.vocab_size, pretrained=True)
		self.attention = Attention(self.channels,self.channels)
		self.rnn = nn.GRU(self.channels,self.channels,1,batch_first=True)
		self.softmax = nn.Softmax(1)
		self.max_len = options['max_len']
	def forward(self,x,hn):
		x,x_weights = self.attention(x,hn)
		hn = hn[:,0].unsqueeze(1)
		x,hn = self.rnn(x,hn.transpose(0,1))
		return x
	def infer(self,features,caption):
		words = torch.zeros(self.max_len,dtype=torch.long)
		for i in range(self.max_len):
			x,x_weights = self.attention(features,caption)
			caption = caption.transpose(0,1)
			caption,hn = self.rnn(features,caption)
			idx = self.word_decoder(caption)
			words[i] = idx
		return words

class Image2Caption(nn.Module):
	def __init__(self):
		super(Image2Caption, self).__init__()
		self.channels = options['output_channels']
		self.vocab_size = options['vocab_size']
		self.embedding = nn.Embedding(self.vocab_size, self.channels)
		self.encoder = ImageEncoder()
		self.decoder = CaptionDecoder()
		self.LR = options['learning_rate']
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR,
										  weight_decay=1e-5)
	def forward(self, image, captions):
		features = self.encoder(image)
		input_captions = self.embedding(captions)
		output_captions = self.decoder(features,input_captions)
		return output_captions, input_captions.detach()
	def infer(self, image):
		features = self.encoder(image)
		caption = self.embedding(torch.zeros(1,dtype=torch.long))
		captions = self.decoder.infer(features, caption)
		return captions
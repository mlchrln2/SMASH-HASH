#dependencies
import torch
import torch.nn as nn
import torchvision.models as models
from Attention import LocalAttention2d
from torch.nn.utils.rnn import pack_padded_sequence

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
		self.align_size = options['align_size']
		self.window = options['window']
		#self.image_encoder = ImageEncoder(channel_size=self.channel_size)
		self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
									  embedding_dim=self.embed_size)
		self.bi_factor = (2 if self.bidirectional else 1)
		self.attention = LocalAttention2d(query_size=self.channel_size,
								   		  context_size=self.embed_size,
								   		  align_size=self.align_size,
								   		  window=self.window)
		self.rnn = nn.GRU(input_size=self.channel_size,
						  hidden_size=self.hidden_size,
						  bidirectional=self.bidirectional,
						  num_layers=self.num_layers,
						  batch_first=True)
		self.decoder = nn.Linear(in_features=self.hidden_size*self.bi_factor, 
								 out_features=self.vocab_size)
		self.optimizer = torch.optim.Adam(params=self.parameters(), 
										  lr=self.LR,
										  weight_decay=1e-5)
		self.criterion = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(1)
	def forward(self, images, captions, lengths):
		#features = self.image_encoder(images)
		features = images
		captions = self.embedding(captions)
		z_inputs = self.attention(features,captions)
		captions = pack_padded_sequence(z_inputs,lengths,batch_first=True)
		z_inputs,_ = self.rnn(z_inputs)
		captions = self.decoder(z_inputs)
		captions = self.softmax(captions.transpose(1,2))
		return captions
	def infer(self, image):
		#features = self.image_encoder(image)
		features = image
		idxs = torch.zeros(1,dtype=torch.long)
		hn = None
		words = torch.zeros(self.max_len,dtype=torch.long)
		frames = torch.zeros(self.max_len,4,self.window[0],self.window[0])
		num_words = 0
		for num_words in range(self.max_len):
			caption = self.embedding(idxs.unsqueeze(0))
			z_inputs,frame = self.attention.infer(features,caption)
			frames[num_words] = frame
			z_inputs,hn = self.rnn(z_inputs,hn)
			caption = self.decoder(z_inputs.squeeze(1))
			idxs = torch.argmax(caption,1)
			words[num_words] = idxs
			if idxs.item() == 1:
				break
		frames = frames[0].unsqueeze(0) if num_words == 0 else frames[:num_words]
		frames = frames.permute(1,2,0,3).contiguous()
		frames = frames.view(frames.size(0), frames.size(1),frames.size(2)*frames.size(3))
		words = words[0].unsqueeze(0) if num_words == 0 else words[:num_words]
		hn = hn.transpose(0,1).contiguous()
		summaries = hn.view(self.hidden_size*self.bi_factor)
		return words, summaries, frames

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
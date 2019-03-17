#dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Attention import LocalAttention2d
from pack_padded_sequence import pack_padded_sequence

#user defined modules
from HyperParameters import options

class Image2Caption(nn.Module):
	def __init__(self):
		super(Image2Caption, self).__init__()
		self.channel_size = options['channel_size']
		self.embed_size = options['embed_size']
		self.vocab_size = options['vocab_size']
		self.max_len = options['max_len']
		self.LR = options['learning_rate']
		self.hidden_size = options['hidden_size']
		self.window = options['window']
		self.drop = options['drop']
		self.image_encoder = ImageEncoder(channel_size=self.channel_size)
		self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
									  embedding_dim=self.embed_size)
		self.dropout = nn.Dropout(self.drop)
		self.attention = LocalAttention2d(query_size=self.channel_size,
								   		  context_size=self.hidden_size,
								   		  window=self.window)
		self.rnn = nn.GRUCell(input_size=self.channel_size+self.embed_size,
							  hidden_size=self.hidden_size)
		self.decoder = nn.Linear(in_features=self.hidden_size, 
								 out_features=self.vocab_size)
		self.decoder_dropout = nn.Dropout(self.drop)
		self.optimizer = torch.optim.Adam(params=self.parameters(),
										  lr=self.LR)
		self.criterion = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(1)
	def forward(self, images, captions, lengths):
		features = self.image_encoder(images)
		captions = self.embedding(captions)
		captions = self.dropout(captions)
		captions = captions.transpose(0,1)
		hn = torch.zeros(features.size(0),self.hidden_size)
		outputs = torch.zeros(features.size(0),captions.size(0),self.vocab_size)
		batch = len(lengths)
		batch_item = lengths[batch-1].item()
		for i,cap in enumerate(captions):
			while i+1 > batch_item and batch > 1:
				batch -= 1
				batch_item = lengths[batch-1].item()
			cap = cap[:batch]
			hn = hn[:batch]
			feats = self.attention(features[:batch],hn)
			z_inputs = torch.cat((feats,cap),1)
			hn = self.rnn(z_inputs,hn)
			out = self.decoder_dropout(hn)
			outputs[:batch,i] = self.decoder(out)
		return outputs
	def infer(self, image):
		features = self.image_encoder(image)
		idxs = torch.zeros(1,dtype=torch.long)
		num_words = 0
		words = torch.zeros(self.max_len,dtype=torch.long)
		alphas = torch.zeros(features.size(0),self.max_len,features.size(2),features.size(3))
		hn = torch.zeros(features.size(0),self.hidden_size)
		for num_words in range(self.max_len):
			if idxs.item() == 1:
				break
			feats,alpha = self.attention.infer(features,hn)
			cap = self.embedding(idxs)
			z_inputs = torch.cat((feats,cap),1)
			hn = self.rnn(z_inputs,hn)
			output = self.decoder(hn)
			idxs = torch.argmax(output,1)
			words[num_words] = idxs
			alphas[:,num_words] = alpha
		alphas = alphas[:,0].unsqueeze(1) if num_words == 0 else alphas[:,:num_words]
		alphas = F.interpolate(alphas, size=(image.size(2),image.size(3)), mode='nearest')
		words = words[0].unsqueeze(0) if num_words == 0 else words[:num_words]
		summaries = hn
		return words, summaries, alphas

class ImageEncoder(nn.Module):
	def __init__(self, channel_size):
		super(ImageEncoder,self).__init__()
		self.channel_size = channel_size
		pretrained_net = models.vgg16(pretrained=True).features
		modules = list(pretrained_net.children())[:29]
		self.pretrained_net = nn.Sequential(*modules)

		self.bn = nn.BatchNorm2d(num_features=modules[-1].in_channels,
								 momentum=0.01)
	def forward(self, x):
		with torch.no_grad():
			x = self.pretrained_net(x)
		x = self.bn(x)
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
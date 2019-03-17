#dependencies
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import nltk
import gc
import os
import h5py
import numpy as np

#user defined modules
from HyperParameters import options

class CocoCaptions(data.Dataset):
	def __init__(self, root, annFile, idx2word_file, target_transform=None):
		from pycocotools.coco import COCO
		self.root = os.path.expanduser(root)
		self.coco = COCO(annFile)
		self.ids = list(self.coco.imgs.keys())
		self.target_transform = target_transform
		self.idx2word_file = idx2word_file
		self.idx2word = h5py.File(self.idx2word_file, 'w')
		self.itr = 3
		self.idx2word.create_dataset(name=str(0),data=np.string_(options['start_word']))
		self.idx2word.create_dataset(name=str(1),data=np.string_(options['end_word']))
		self.idx2word.create_dataset(name=str(2),data=np.string_(options['unk_word']))
	def __getitem__(self, index):
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		anns = [anns[0]]
		for ann in anns:
			for word in nltk.tokenize.word_tokenize(str(ann['caption']).lower()):
				if not word in word2idx.keys():
					self.idx2word.create_dataset(name=str(self.itr),data=np.string_(word))
					word2idx[word] = self.itr
					self.itr += 1
		return self.itr
	def __len__(self):
		return len(self.ids)

#dataset location in memory
data_dir = options['data_dir']

#transform images from PIL images to tensors
transform = transforms.ToTensor()

#load data and captions in batches
dataset = CocoCaptions(root='{}/train2017/'.format(data_dir), 
					   annFile='{}/annotations/captions_train2017.json'.format(data_dir),
					   idx2word_file='{}/idx2word.h5'.format(data_dir))
dataloader = DataLoader(dataset=dataset,
						batch_size=1,
						shuffle=False,
						drop_last=True)

gc.collect()
word2idx = {}

start_word = options['start_word']
end_word = options['end_word']
unk_word = options['unk_word']
word2idx[start_word] = 0
word2idx[end_word] = 1
word2idx[unk_word] = 2

print('loading dictionary...')
for itr in dataloader:
	print('{} words loaded into vocabulary'.format(itr.item()+1), end='\r')
print('{} words loaded into vocabulary'.format(itr.item()+1))
print('complete')
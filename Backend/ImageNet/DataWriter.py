#dependencies
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import gc
import os
import h5py
import nltk

#user defined modules
from HyperParameters import options
from Vocabulary import word2idx
from Vocabulary import idx2word


class AnnotationWriter(data.Dataset):
	def __init__(self, root, annFile, filename, target_transform=None):
		from pycocotools.coco import COCO
		self.root = os.path.expanduser(root)
		self.coco = COCO(annFile)
		self.filename = filename
		self.coco_dataset = h5py.File(self.filename, 'w')
		self.ids = list(self.coco.imgs.keys())
		self.target_transform = target_transform
		self.start_word = options['start_word']
		self.end_word = options['end_word']

	def __getitem__(self, index):
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		counts = torch.tensor([[i+1,len(nltk.tokenize.word_tokenize(str(ann['caption']).lower()))] for i,ann in enumerate(anns)])
		(max_i,max_j),_ = torch.max(counts,0)
		counts = counts[:,1].unsqueeze(1)
		targets = torch.zeros(max_i, max_j+2,dtype=torch.long)
		for i, ann in enumerate(anns):
			sentence = nltk.tokenize.word_tokenize(str(ann['caption']).lower())
			idx = 0
			for j in range(max_j+2):
				if j == 0:
					idx = word2idx[self.start_word]
				elif j < len(sentence)+1:
					idx = word2idx[sentence[j-1]]
				else:
					idx = word2idx[self.end_word]
				targets[i,j] = idx
		if self.target_transform is not None:
			targets = self.target_transform(targets)
		targets = torch.cat((targets,counts),1)
		self.coco_dataset.create_dataset(name=str(index),shape=targets.size(),dtype='u4',data=targets)
		return targets
	def __len__(self):
		return len(self.ids)

#dataset location in memory
data_dir = options['data_dir']

#load data and captions in batches
dataset = AnnotationWriter(root='{}/train2017/'.format(data_dir),
						   annFile='{}/annotations/captions_train2017.json'.format(data_dir),
						   filename='{}/coco_annotations.h5'.format(data_dir),
						   )

for i,_ in enumerate(dataset):
	print('iteration {} of {}'.format(i, len(dataset)),end='\r')
print('complete')
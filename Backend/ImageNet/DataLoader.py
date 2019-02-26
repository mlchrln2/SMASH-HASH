import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path

from HyperParameters import options
from Vocabulary import word2idx
from Vocabulary import idx2word

class CocoDataset(data.Dataset):
	"""`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

	Args:
		root (string): Root directory where images are downloaded to.
		annFile (string): Path to json annotation file.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.ToTensor``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
	"""
	def __init__(self, root, annFile, transform=None, target_transform=None):
		from pycocotools.coco import COCO
		self.root = os.path.expanduser(root)
		self.coco = COCO(annFile)
		self.ids = list(self.coco.imgs.keys())
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: Tuple (image, target). target is a list of captions for the image.
		"""
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)

		(max_i,max_j) = (0,0)
		for max_i, ann in enumerate(anns):
			size = len(ann['caption'])
			if size > max_j:
				max_j = size

		target = torch.zeros(max_i+1, max_j+1,dtype=torch.long)
		for i, ann in enumerate(anns):
			sentence = ann['caption'].split()
			idx = 0
			for j in range(max_j+1):
				if j == 0:
					idx = word2idx['START']
				elif j < len(sentence)+1:
					idx = word2idx[sentence[j-1]]
				else:
					idx = word2idx['STOP']
				target[i,j] = idx


		path = coco.loadImgs(img_id)[0]['file_name']
		img = Image.open(os.path.join(self.root, path)).convert('RGB')

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.ids)

def collate(data):
	'''pads the data with 0's'''
	images, captions = zip(*data)
	length = torch.tensor([[item.size(1),item.size(2)] for item in images])
	(max_i,max_j),_ = torch.max(length,0)
	images = [nn.ZeroPad2d((0, max_j.item()-item.size(2), 0, max_i.item()-item.size(1)))
			(item) for item in images]
	images = torch.stack(images, 0)

	length = torch.tensor([[item.size(0),item.size(1)] for item in captions])
	(max_i,max_j),_ = torch.max(length,0)
	captions = [nn.ZeroPad2d((0, max_j.item()-item.size(1), 0, max_i.item()-item.size(0)))
			   (item) for item in captions]
	captions = torch.stack(captions, 0)
	return images, captions

#transform for data
transform = transforms.ToTensor()

#dataset location in memory
myPassport_dir = '/media/orlandomelchor/My Passport/datasets/coco-dataset'

#load data and captions in batches
dataset = CocoDataset(root='{}/train2017/'.format(myPassport_dir), 
					  annFile='{}/annotations/captions_train2017.json'.format(myPassport_dir),
					  transform=transform,
					  )

#number of batches in one load
batch_size = options['batch_size']

#batch data loader
dataloader = DataLoader(dataset=dataset, 
						batch_size=batch_size,
						collate_fn=collate,
						shuffle=True,
						drop_last=True)
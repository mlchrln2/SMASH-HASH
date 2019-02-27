#dependencies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
import h5py
import numpy as np

#user defined modules
from HyperParameters import options

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
	def __init__(self, root, annFile, filename, transform=None, target_transform=None):
		from pycocotools.coco import COCO
		self.root = os.path.expanduser(root)
		self.coco = COCO(annFile)
		self.ids = list(self.coco.imgs.keys())
		self.filename = filename
		self.coco_dataset = h5py.File(self.filename, 'r')
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
		path = coco.loadImgs(img_id)[0]['file_name']
		img = Image.open(os.path.join(self.root, path)).convert('RGB')
		targets = self.coco_dataset[str(index)][:].astype(int)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			targets = self.target_transform(targets)
		captions = targets[:,:-1]
		lengths = targets[:,-1]
		return img, captions, lengths

	def __len__(self):
		return len(self.ids)

def collate(batch):
	'''pads the data with 0's'''
	images, captions,lengths = zip(*batch)
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
	return images, captions, lengths

#transform for data

#transform = transforms.ToTensor()
transform = transforms.Compose([ 
    #transforms.Resize(256),                          # smaller edge of image resized to 256
    #transforms.RandomCrop(224),                      # get 224x224 crop from random location
    #transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor()#,                           # convert the PIL Image to a tensor
    #transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
    #                     (0.229, 0.224, 0.225))
    ])
target_transform = torch.from_numpy

#dataset location in memory
myPassport_dir = '/media/orlandomelchor/My Passport/datasets/coco-dataset'


#load data and captions in batches
dataset = CocoDataset(root='{}/train2017/'.format(myPassport_dir), 
					  annFile='{}/annotations/captions_train2017.json'.format(myPassport_dir),
					  filename='{}/coco_annotations.h5'.format(myPassport_dir),
					  transform=transform,
					  target_transform=target_transform
					  )

#number of batches in one load
batch_size = options['batch_size']

#batch data loader
dataloader = DataLoader(dataset=dataset, 
						batch_size=batch_size,
						collate_fn=collate,
						shuffle=True,
						drop_last=True)
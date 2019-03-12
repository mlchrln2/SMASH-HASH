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

#user defined modules
from HyperParameters import options

idx2word_file = 'idx2word.h5'
idx2word = h5py.File(idx2word_file, 'r')

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
		target_loc = self.coco_dataset[str(index)]
		rand_idx = torch.randint(target_loc.shape[0],size=(1,)).item()
		targets = torch.from_numpy(target_loc[rand_idx].astype(int))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			targets = self.target_transform(targets)
		lengths = targets[-1]+2
		captions = targets[:(lengths)]
		return img, captions, lengths

	def __len__(self):
		return len(self.ids)

def collate(batch):
	'''pads the data with 0's'''
	images, captions,lengths = zip(*batch)
	images = torch.stack(images, 0)
	lengths = torch.stack(lengths,0)
	max_j = torch.max(lengths)
	captions = [nn.ConstantPad1d((0, max_j.item()-item.size(0)),1)
			   (item) for item in captions]
	captions = torch.stack(captions, 0)
	order = torch.flip(torch.argsort(lengths),(0,))
	images = images[order]
	captions = captions[order]
	lengths = lengths[order]
	return images, captions, lengths

#transform for data

transform = transforms.Compose([ 
	transforms.Resize(256),                          # smaller edge of image resized to 256
	transforms.RandomCrop(224),                      # get 224x224 crop from random location
	transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
	transforms.ToTensor(),                           # convert the PIL Image to a tensor
	transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
						 (0.229, 0.224, 0.225))
])

#dataset location in memory
data_dir = options['data_dir']

#load data and captions in batches
dataset = CocoDataset(root='{}/train2017/'.format(data_dir), 
					  annFile='{}/annotations/captions_train2017.json'.format(data_dir),
					  filename='{}/coco_annotations.h5'.format(data_dir),
					  transform=transform)

#number of batches in one load
batch_size = options['batch_size']

#batch data loader
train_loader = DataLoader(dataset=dataset, 
						batch_size=batch_size,
						collate_fn=collate,
						shuffle=True,
						drop_last=True)

#1 batch data loader
test_loader = DataLoader(dataset=dataset, 
						batch_size=1,
						collate_fn=collate,
						shuffle=False,
						drop_last=True)
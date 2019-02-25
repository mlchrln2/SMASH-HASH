import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import gc
import os

class CocoCaptions(data.Dataset):
	def __init__(self, root, annFile, target_transform=None):
		from pycocotools.coco import COCO
		self.root = os.path.expanduser(root)
		self.coco = COCO(annFile)
		self.ids = list(self.coco.imgs.keys())
		self.target_transform = target_transform

	def __getitem__(self, index):
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		target = [ann['caption'] for ann in anns]
		if self.target_transform is not None:
			target = self.target_transform(target)
		return target

	def __len__(self):
		return len(self.ids)

#dataset location in memory
myPassport_dir = '/media/orlandomelchor/My Passport/datasets/coco-dataset'

#transform = transforms.ToTensor()
transform = transforms.transforms.ToTensor()

#load data and captions in batches
dataset = CocoCaptions(root='{}/train2017/'.format(myPassport_dir), 
					   annFile='{}/annotations/captions_train2017.json'.format(myPassport_dir))
dataloader = DataLoader(dataset=dataset, 
						batch_size=1,
						shuffle=False,
						drop_last=True)

gc.collect()
word2idx = {}
idx2word = {}
word2idx[''] = 0
idx2word[0] = ''
itr = 1
print('loading dictionary...')
for i,captions in enumerate(dataloader):
	for caption in captions:
		for sentence in caption:
			for word in sentence.split(' '):
				if word not in word2idx.keys():
					word2idx[word] = itr
					idx2word[itr] = word
					itr += 1
	print('{} words loaded into vocabulary'.format(itr), end='\r')
print()
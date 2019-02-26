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
		self.itr = 2

	def __getitem__(self, index):
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		for ann in anns:
			for word in ann['caption'].split():
				if word not in word2idx.keys():
					word2idx[word] = self.itr
					idx2word[self.itr] = word
					self.itr += 1
		return self.itr
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
word2idx['START'] = 0
word2idx['STOP'] = 1
idx2word[0] = 'START'
idx2word[1] = 'STOP'

print('loading dictionary...')
for itr in dataloader:
	print('{} words loaded into vocabulary'.format(itr.item()), end='\r')
print()
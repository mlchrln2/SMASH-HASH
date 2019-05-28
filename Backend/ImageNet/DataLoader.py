"""This module is used to load the training and test datasets for MS COCO Captions"""

import os
import os.path
from torchvision import transforms
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import h5py
from PIL import Image

# user defined modules
from HyperParameters import OPTIONS


class CocoTrain(data.Dataset):
    """MS Coco Captions Training Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
        normalize (callable, optional): A function/transform that takes in a torch.Tensor
                and returns a normalized version.
        target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, annFile, filename, transform=None,
                 normalize=None, target_transform=None, device='cpu'):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.filename = filename
        self.coco_dataset = h5py.File(self.filename, 'r', libver='latest', swmr=True)
        self.transform = transform
        self.normalize = normalize
        self.target_transform = target_transform
        self.device = device

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (img, captions, lengths). lengths is the size of a given caption
            and image is the transformed and normalized image
        """
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        target_loc = self.coco_dataset[str(index)]
        rand_idx = torch.randint(target_loc.shape[0], size=(1,)).item()
        targets = torch.from_numpy(target_loc[rand_idx].astype(int)).to(self.device)
        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        lengths = targets[-1] + 2
        lengths.requires_grad = False
        captions = targets[:(lengths)]
        return img, captions, lengths

    def __len__(self):
        return len(self.ids)


class CocoTest(data.Dataset):
    """MS Coco Captions Testing Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
        normalize (callable, optional): A function/transform that takes in a torch.Tensor
                and returns a normalized version.
        target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, annFile, filename, transform=None,
                 normalize=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.filename = filename
        self.coco_dataset = h5py.File(self.filename, 'r')
        self.transform = transform
        self.normalize = normalize
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Inputs:
            index (int): Index

        Outputs:
            tuple: Tuple (norm_image, img, captions, lengths).
            lengths is the size of a given caption. norm_img is used for model inference while
            img is used for visualization.
        """
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        target_loc = self.coco_dataset[str(index)]
        rand_idx = 0
        targets = torch.from_numpy(target_loc[rand_idx].astype(int))
        norm_img = img
        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            norm_img = self.normalize(img.clone())
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        lengths = targets[-1] + 2
        lengths.requires_grad = False
        captions = targets[:(lengths)]
        return norm_img, img, captions, lengths

    def __len__(self):
        return len(self.ids)


class CocoVal(data.Dataset):
    """MS Coco Captions Validation Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
        normalize (callable, optional): A function/transform that takes in a torch.Tensor
                and returns a normalized version.
        target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, annFile, filename, transform=None,
                 normalize=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.filename = filename
        self.coco_dataset = h5py.File(self.filename, 'r')
        self.transform = transform
        self.normalize = normalize
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (img, captions, lengths). lengths is the size of a given caption
            and image is the transformed and normalized image
        """
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        target_loc = self.coco_dataset[str(index)]
        rand_idx = torch.randint(target_loc.shape[0], size=(1,)).item()
        targets = torch.from_numpy(target_loc[rand_idx].astype(int))
        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        lengths = targets[-1] + 2
        lengths.requires_grad = False
        captions = targets[:(lengths)]
        return img, captions, lengths

    def __len__(self):
        return len(self.ids)


def collate(batch):
    """Stacks the images, pads the captions with 1's (end_word) and orders the lengths
    in decending order.

    Inputs:
        batch (tuple): Loaded images, captions, and lengths for the current batch

    Outputs:
        tuple: Tuple (norm_image, img, captions, lengths). lengths is the size of a given caption.
        norm_img is used for model inference while img is used for visualization.
    """
    images, captions, lengths = zip(*batch)
    images = torch.stack(images, 0)
    lengths = torch.stack(lengths, 0)
    max_j = torch.max(lengths)
    captions = [nn.ConstantPad1d((0, max_j.item() - item.size(0)), 1)
                (item) for item in captions]
    captions = torch.stack(captions, 0)
    order = torch.flip(torch.argsort(lengths), (0,))
    images = images[order]
    captions = captions[order]
    lengths = lengths[order]
    return images, captions, lengths

# dataset location in memory
DATA_DIR = OPTIONS['data_dir']

# indicate which device to train data on
DEVICE = OPTIONS['device']

# transforms for data
TRAIN_TRANSFORM = transforms.Compose([
    # smaller edge of image resized to 256
    transforms.Resize(256),
    # get 224x224 crop from random location
    transforms.RandomCrop(224),
    # horizontally flip image with probability=0.5
    transforms.RandomHorizontalFlip(),
    # convert the PIL Image to a tensor
    transforms.ToTensor(),
])

TEST_TRANSFORM = transforms.Compose([
    # smaller edge of image resized to 256
    transforms.Resize(256),
    # get 224x224 crop from random location
    transforms.CenterCrop(224),
    # convert the PIL Image to a tensor
    transforms.ToTensor(),
])

# transforms for data
VAL_TRANSFORM = transforms.Compose([
    # smaller edge of image resized to 256
    transforms.Resize(256),
    # get 224x224 crop from random location
    transforms.CenterCrop(224),
    # convert the PIL Image to a tensor
    transforms.ToTensor(),
])

# normalizers for data
TRAIN_NORMALIZE = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

TEST_NORMALIZE = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

# normalizers for data
VAL_NORMALIZE = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

# load data and captions in batches
TRAIN_DATASET = CocoTrain(root='{}/train2017/'.format(DATA_DIR),
                          annFile='{}/annotations/captions_train2017.json'.format(
                              DATA_DIR),
                          filename='{}/train_captions.h5'.format(DATA_DIR),
                          transform=TRAIN_TRANSFORM,
                          normalize=TRAIN_NORMALIZE,
                          device=DEVICE)

# load data and captions in batches
TEST_DATASET = CocoTest(root='{}/train2017/'.format(DATA_DIR),
                        annFile='{}/annotations/captions_train2017.json'.format(
                            DATA_DIR),
                        filename='{}/train_captions.h5'.format(DATA_DIR),
                        transform=TEST_TRANSFORM,
                        normalize=TEST_NORMALIZE)

VAL_DATASET = CocoVal(root='{}/val2017/'.format(DATA_DIR),
                      # load data and captions in batches
                      annFile='{}/annotations/captions_val2017.json'.format(
                          DATA_DIR),
                      filename='{}/val_captions.h5'.format(DATA_DIR),
                      transform=VAL_TRANSFORM,
                      normalize=VAL_NORMALIZE)

# load the idx2word dictionary
IDX2WORD = h5py.File('{}/idx2word.h5'.format(DATA_DIR), 'r')

# batch data loader
TRAIN_LOADER = DataLoader(dataset=TRAIN_DATASET,
                          batch_size=OPTIONS['batch_size'],
                          collate_fn=collate,
                          shuffle=True,
                          drop_last=True)

# 1 batch data loader
TEST_LOADER = DataLoader(dataset=TEST_DATASET,
                         batch_size=1,
                         shuffle=False,
                         drop_last=True)

# 1 batch data loader
VAL_LOADER = DataLoader(dataset=VAL_DATASET,
                        batch_size=1,
                        collate_fn=collate,
                        shuffle=False,
                        drop_last=True)

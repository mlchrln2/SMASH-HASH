'''This module builds the word-to-index mapping used by the DataWriter.py file and creates a
dictionary for the index-to-word mapping stored for easy translation during model
evaluation phase.'''

# dependencies
import gc
import os
import nltk
import h5py
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

# user defined modules
from HyperParameters import OPTIONS


class Vocabulary(data.Dataset):
    """MS Coco Vocabulary Dataset.

    Args:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            idx2word_file (string): Path to idx2word dictionary file.
            target_transform (callable, optional): A function/transform that takes in the
                    target and transforms it.
    """

    def __init__(self, root, annFile, idx2word_file, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.target_transform = target_transform
        self.idx2word_file = idx2word_file
        self.idx2word = h5py.File(self.idx2word_file, 'w')
        self.itr = 3
        self.idx2word.create_dataset(
            name=str(0), data=np.string_(OPTIONS['start_word']))
        self.idx2word.create_dataset(
            name=str(1), data=np.string_(OPTIONS['end_word']))
        self.idx2word.create_dataset(
            name=str(2), data=np.string_(OPTIONS['unk_word']))

    def __getitem__(self, index):
        """
        Creates an index-to-word file that contains all of unique captions that are tied to
        an image. Only the first annotation is used to greatly decrease the size of
        the vocabulary. The file created is reference when validating the results of the model.

        Inputs:
            index (int): Index

        Outputs:
            itr: the number of unique words loaded so far into the dictionary file.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        anns = [anns[0]]
        for ann in anns:
            for word in nltk.tokenize.word_tokenize(str(ann['caption']).lower()):
                if not word in WORD2IDX.keys():
                    self.idx2word.create_dataset(
                        name=str(self.itr), data=np.string_(word))
                    WORD2IDX[word] = self.itr
                    self.itr += 1
        return self.itr

    def __len__(self):
        return len(self.ids)

# dataset location in memory
DATA_DIR = OPTIONS['data_dir']

# load data and captions from memory
DATASET = Vocabulary(root='{}/train2017/'.format(DATA_DIR),
                     annFile='{}/annotations/captions_train2017.json'.format(
    DATA_DIR),
    idx2word_file='{}/idx2word.h5'.format(DATA_DIR))

# load data and captions in batches
DATALOADER = DataLoader(dataset=DATASET,
                        batch_size=1,
                        shuffle=False,
                        drop_last=True)

'''fill the first three indices of WORD2IDX with the start, end, and unknown word labels'''
WORD2IDX = {}
WORD2IDX[OPTIONS['start_word']] = 0
WORD2IDX[OPTIONS['end_word']] = 1
WORD2IDX[OPTIONS['unk_word']] = 2

'''Parse through all the unique words in the captions and add them to the idx2word.h5 file
if they are unique'''
print('loading dictionary...')
ITR = 0
gc.collect()

for ITR in DATALOADER:
    print('{} words loaded into vocabulary'.format(ITR.item() + 1), end='\r')
print('{} words loaded into vocabulary'.format(ITR.item() + 1))
print('complete')

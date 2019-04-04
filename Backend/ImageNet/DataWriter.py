'''This module creates the COCO annotation file to facilitate the training procedure
and calls Vocabulary.py'''

# dependencies
import gc
import os
import torch
import torch.utils.data as data
import h5py
import nltk

# user defined modules
from HyperParameters import OPTIONS
from Vocabulary import word2idx


class AnnotationWriter(data.Dataset):
    """MS Coco Captions Training Dataset.

    Args:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            transform (callable, optional): A function/transform that takes in an PIL image
                    and returns a transformed version. E.g, ``transforms.ToTensor``
            target_transform (callable, optional): A function/transform that takes in the
                    target and transforms it.
    """

    def __init__(self, root, annFile, filename, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.filename = filename
        self.coco_dataset = h5py.File(self.filename, 'w')
        self.ids = list(self.coco.imgs.keys())
        self.target_transform = target_transform
        self.start_word = OPTIONS['start_word']
        self.end_word = OPTIONS['end_word']

    def __getitem__(self, index):
        """
		Creates a coco annotation file that contains all of the captions that are tied to
		an image. Only the first annotation is considered to greatly decrease the size of
		the vocabulary. Any upper case words are all treated as lower case. In order to
		avoid converting words to indices while training it is done here and saved to a file.

        Inputs:
            index (int): Index

        Outputs:
            targets: unimportant parameter that is ultimately ignored
        """

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        anns = [anns[0]]
        counts = torch.tensor(
            [[i + 1, len(nltk.tokenize.word_tokenize(str(ann['caption']).lower()))]
             for i, ann in enumerate(anns)])
        (max_i, max_j), _ = torch.max(counts, 0)
        counts = counts[:, 1].unsqueeze(1)
        targets = torch.zeros(max_i, max_j + 2, dtype=torch.long)
        for j, ann in enumerate(anns):
            sentence = nltk.tokenize.word_tokenize(str(ann['caption']).lower())
            idx = 0
            for k in range(max_j + 2):
                if k == 0:
                    idx = word2idx[self.start_word]
                elif k < len(sentence) + 1:
                    idx = word2idx[sentence[k - 1]]
                else:
                    idx = word2idx[self.end_word]
                targets[j, k] = idx
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        targets = torch.cat((targets, counts), 1)
        self.coco_dataset.create_dataset(
            name=str(index), shape=targets.size(), dtype='u4', data=targets)
        return targets

    def __len__(self):
        return len(self.ids)

# dataset location in memory
DATA_DIR = OPTIONS['data_dir']

# load data and captions in batches
DATASET = AnnotationWriter(root='{}/train2017/'.format(DATA_DIR),
                           annFile='{}/annotations/captions_train2017.json'.format(
                               DATA_DIR),
                           filename='{}/coco_annotations.h5'.format(DATA_DIR),
                           )
gc.collect()
i = 0
# write data and captions
for i, _ in enumerate(DATASET):
    print('annotation {} of {} loaded'.format(i, len(DATASET)), end='\r')
print('annotation {} of {} loaded'.format(i, len(DATASET)))
print('complete')

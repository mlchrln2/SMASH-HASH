'''Basic method for visualizing the frequency of words in the COCO captions'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
from HyperParameters import OPTIONS

# load all of the hyperparameters
DATA_DIR = OPTIONS['data_dir']

# load the coco annotation file
FILENAME = '{}/coco_annotations.h5'.format(DATA_DIR)
COCO_DATASET = h5py.File(FILENAME, 'r')

# retrieve all the keys from the dataset
SIZE = len(COCO_DATASET.keys())

# increment the values of the keys every time they appear in the file
DICTION = {}
for i in range(SIZE):
    target_loc = COCO_DATASET[str(i)]
    for j, _ in enumerate(target_loc):
        for word in target_loc[j]:
            if word not in DICTION:
                DICTION[word] = 1
            else:
                DICTION[word] += 1
    print('{} of {}'.format(i, SIZE), end='\r')

# retrieve the indices and the frequency of those indices in the file
KEYS = np.array(list(DICTION.keys()))
VALUES = np.array(list(DICTION.values()))

'''plot the index vs frequency plot as long as the there are at least thresh
items in the dataset'''
for thresh in range(100):
    plt.title(thresh)
    plt.scatter(KEYS[VALUES > thresh], VALUES[VALUES > thresh])
    plt.show()

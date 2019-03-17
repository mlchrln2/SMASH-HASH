import h5py
from HyperParameters import options
import numpy as np
import matplotlib.pyplot as plt

data_dir = options['data_dir']
filename = '{}/coco_annotations.h5'.format(data_dir)
coco_dataset = h5py.File(filename,'r')
size = len(coco_dataset.keys())
d = {}
for i in range(size):
	target_loc = coco_dataset[str(i)]
	for j in range(len(target_loc)):
		for word in target_loc[j]:
			if word not in d:
				d[word] = 1
			else:
				d[word] += 1
	print('{} of {}'.format(i,size),end='\r')

keys = np.array(list(d.keys()))
values = np.array(list(d.values()))
for thresh in range(100):
	plt.title(thresh)
	plt.scatter(keys[values>thresh],values[values>thresh])
	plt.show()
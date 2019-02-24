import h5py
import os
import numpy as np

#dataset location in memory
myPassport_dir = '/media/orlandomelchor/My Passport/datasets/coco-dataset/'
data_dir = 'train2017/'

files = np.array([myPassport_dir+data_dir+file for file in os.listdir(myPassport_dir+data_dir)])

filename = 'num_to_image_dict.h5'
hf = h5py.File(filename, 'w')
dictionary = {}
for i,file in enumerate(files):
	dictionary[i] = file
	file = file.split('/')[-1]
	hf.create_dataset(str(i),data=np.string_(file))
	print("key {} of {}".format(i,len(files)),end='\r')
hf.close()

filename = 'test_file.h5'
hf = h5py.File(filename, 'w')

nodes = np.arange(len(dictionary))
np.random.shuffle(nodes)
nodes = nodes[:20]

for i, node in enumerate(nodes):
	connections = nodes[nodes!=node].copy()
	np.random.shuffle(connections)
	num_connections = np.random.randint(len(connections),size=1)[0]
	connections = connections[:num_connections]
	weights = np.random.random(len(connections))
	data = np.vstack((connections,weights))
	hf.create_dataset(str(node),data=data)
hf.close()

output = 'cp '
for i, node in enumerate(nodes):
	output+=dictionary[node] + ' '
output+= './'
print(output)

# coding: utf-8

# In[1]:


from pyvis.network import Network
#import networkx as nx
import ipywidgets as wd
import numpy as np
import h5py
import os
from IPython.display import Image

##current working directory
cwd = os.getcwd()

##main network graph
net = Network()

##dictionary
hf = h5py.File(cwd + '\\dummy-data\\num_to_image_dict.h5', 'r')
keys = list(hf.keys())

##adjacency list
data_file = h5py.File(cwd + '\\dummy-data\\test_file.h5', 'r')
dataKeys = list(data_file.keys())


# In[2]:


##storage for vertex to check duplicates
vertex_store = []

##node id
id=0

##generate vertices
for i in range(len(dataKeys)):
    string = list(data_file.values())[i].name
    num = int(string[1:])
    if(not num in vertex_store):
        id+=1
        vertex_store.append(num)
        #print(id, ", ",num)
        net.add_node(id, label=None,shape="circularImage", image = cwd + "\\dummy-data\\" +str(np.core.defchararray.decode(hf[str(num)])))

        
##generate edges (*****not finished yet*****)
for i in range(len(dataKeys)):
    string = list(data_file.values())[i].name
    num = int(string[1:])
    cur_set = data_file[dataKeys[i]]
    labels = []
    
    for elem in cur_set[0]:
        labels.append(elem)
    lables = labels[::-1]
    
    for elem in cur_set[1]:
        #print(elem)
        pass

        
##show network
net.show("mygraph.html")



# In[123]:


##temporary notes
path = cwd + "\\dummy-data\\" +str(np.core.defchararray.decode(hf[str(101869)]))
str(np.core.defchararray.decode(hf[keys[0]]))


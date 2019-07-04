Smash-Hash: Image Recommendation Network
================================================
A two part network that first captions an image and then uses the caption embedding to recommend similar images on a graph. This project is still in development and needs minor improvements to the captioning network and reformulation of the Graphical Convolution Network.

Dependencies:
------------
* torch
* torchvision
* tensorboardX
* nltk
* h5py
* PIL
* gc

ImageNet: Image Captioning Network
================================================
The captioning network first encodes the image using the VGG16 pretrained network and then applies a 2D Attention layer over the encoded image. The 2D Attention layer invented for the purposes of this project is an adapted form of Luong self-attention. The 2D Attention layer extends the reach of Luong self-attention to 2 dimensions making it possible to query images in addition to phrases in NLP. The output from the 2D Attention layer is then concatenated with the previous word caption and fed into an RNN to produce the next most likely word in the caption. The process ends when the 'END' token is obtained and the caption, attention weights, and the encoded sentence are returned.

Data Preparation (for development):
------------
```bash
# To create data files for faster training and validation:
python3 DataWriter.py
```

Parameter Setup (for development):
------------
``` bash
# Edit this file to toggle hyperparameters:
python3 HyperParameters.py
```

Training Setup (for development):
------------
```bash
# To train a new model:
python3 ImageTrain.py

# To continue training an old model:
python3 ImageTrain.py /path/to/file
```

Model Validation (for development):
------------
```bash
# To check the effectiveness of the model with the greedy search algorithm:
python3 ImageVal.py /path/to/file greedy

# To check the effectiveness of the model with the beam search algorithm:
python3 ImageVal.py /path/to/file beam
```
Demonstrations:
------------
```bash
# Check the jupyter notebook for some intermediate results:
jupyter notebook ImageNetDemo.ipynb
```

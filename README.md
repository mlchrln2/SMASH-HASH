Smash-Hash: Image Recommendation Network
================================================
A two part neural network that first captions an image and then uses the caption embeddings to recommend similar images on a graph. This project is still in development and needs minor improvements to the captioning network and reformulation of the Graphical Convolution Network.

Dependencies:
------------
* [torch - 1.1.0](https://pytorch.org/)
* [torchvision - 0.3.0](https://pytorch.org/)
* [tensorboardX - 1.7](https://pypi.org/project/tensorboardX/)
* [nltk - 3.4.1](https://pypi.org/project/nltk/)
* [h5py - 2.9.0](https://pypi.org/project/h5py/)
* [PIL - 6.0.0](https://pypi.org/project/Pillow/)
* [pycocotools - 2.0.0](https://pypi.org/project/pycocotools/)

ImageNet: Image Captioning Network
================================================
The captioning network first encodes the image using the VGG16 pretrained network and then applies a 2D Attention layer over the encoded image. The 2D Attention layer invented for the purposes of this project is an adapted form of [Luong self-attention](https://arxiv.org/pdf/1508.04025.pdf). The 2D Attention layer extends the reach of Luong self-attention to problems in computer vision and instead introduces an image centroid with a gaussian window around that centroid.

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
